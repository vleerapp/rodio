use core::time::Duration;
use std::{
    fmt::{self, Debug},
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};
use symphonia::{
    core::{
        audio::AudioSpec,
        codecs::{
            CodecParameters,
            audio::{AudioDecoder, AudioDecoderOptions, CODEC_ID_NULL_AUDIO},
            registry::CodecRegistry,
        },
        errors::Error,
        formats::{FormatOptions, FormatReader, SeekMode, SeekTo, SeekedTo, TrackType, probe::Hint},
        io::MediaSourceStream,
        meta::MetadataOptions,
        units::{TimeBase, Time},
    },
    default::get_probe,
};

use super::{DecoderError, Settings};
use crate::{
    Source,
    common::{ChannelCount, Sample, SampleRate, assert_error_traits},
    source::{self, padding_samples_needed},
};
use dasp_sample::Sample as _;

#[derive(Clone)]
pub(crate) struct Registry(Arc<RwLock<CodecRegistry>>);

impl Debug for Registry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Registry")
    }
}

impl Registry {
    pub(crate) fn new(registry: CodecRegistry) -> Self {
        Self(Arc::new(RwLock::new(registry)))
    }

    pub(crate) fn write(&self) -> RwLockWriteGuard<'_, CodecRegistry> {
        self.0.write().unwrap()
    }

    pub(crate) fn read(&self) -> RwLockReadGuard<'_, CodecRegistry> {
        self.0.read().unwrap()
    }
}

pub(crate) struct SymphoniaDecoder {
    decoder: Box<dyn AudioDecoder>,
    current_span_offset: usize,
    format: Box<dyn FormatReader>,
    total_duration: Option<Duration>,
    buffer: Vec<Sample>,
    spec: AudioSpec,
    seek_mode: SeekMode,
    selected_track_id: u32,
    time_base: Option<TimeBase>,
    samples_in_current_frame: usize,
    silence_samples_remaining: usize,
    /// True when a prior seek targeted a position past the end of the stream. The iterator
    /// reports no more samples (after flushing any in-flight frame padding) until another
    /// `try_seek` resets the position.
    seeked_past_end: bool,
}

impl SymphoniaDecoder {
    pub(crate) fn new(
        mss: MediaSourceStream<'static>,
        settings: &Settings,
    ) -> Result<Self, DecoderError> {
        match SymphoniaDecoder::init(mss, settings) {
            Err(e) => match e {
                Error::IoError(e) => Err(DecoderError::IoError(e.to_string())),
                Error::DecodeError(e) => Err(DecoderError::DecodeError(e)),
                Error::SeekError(_) => {
                    unreachable!("Seek errors should not occur during initialization")
                }
                Error::Unsupported(_) => Err(DecoderError::UnrecognizedFormat),
                Error::LimitError(e) => Err(DecoderError::LimitError(e)),
                Error::ResetRequired => Err(DecoderError::ResetRequired),
                _ => Err(DecoderError::UnrecognizedFormat),
            },
            Ok(Some(decoder)) => Ok(decoder),
            Ok(None) => Err(DecoderError::NoStreams),
        }
    }

    #[inline]
    pub(crate) fn into_inner(self) -> MediaSourceStream<'static> {
        self.format.into_inner()
    }

    fn init(
        mss: MediaSourceStream<'static>,
        settings: &Settings,
    ) -> symphonia::core::errors::Result<Option<SymphoniaDecoder>> {
        let mut hint = Hint::new();
        if let Some(ext) = settings.hint.as_ref() {
            hint.with_extension(ext);
        }
        if let Some(typ) = settings.mime_type.as_ref() {
            hint.mime_type(typ);
        }
        let format_opts: FormatOptions = Default::default();
        let metadata_opts: MetadataOptions = Default::default();
        let _ = settings.gapless;
        let seek_mode = if settings.coarse_seek {
            SeekMode::Coarse
        } else {
            SeekMode::Accurate
        };
        let mut format = get_probe().probe(&hint, mss, format_opts, metadata_opts)?;

        let track = match format.first_track_known_codec(TrackType::Audio) {
            Some(track) => track,
            None => return Ok(None),
        };
        let track_id = track.id;
        let time_base = track.time_base;
        // `track.duration` is the playable length in timebase units. When gapless playback is
        // disabled the decoder will additionally emit encoder delay/padding frames, so include
        // those in the reported total so it reflects the actual sample count consumers will see.
        let trim_ticks = if settings.gapless {
            0
        } else {
            u64::from(track.delay.unwrap_or(0)) + u64::from(track.padding.unwrap_or(0))
        };
        let total_ticks = track
            .duration
            .map(|d| d.get())
            .or(track.num_frames)
            .map(|n| n.saturating_add(trim_ticks));

        let audio_params = match track.codec_params.as_ref() {
            Some(CodecParameters::Audio(params)) if params.codec != CODEC_ID_NULL_AUDIO => params,
            _ => return Ok(None),
        };

        let decoder_opts = AudioDecoderOptions::default().gapless(settings.gapless);
        let mut decoder = settings
            .codec_registry
            .read()
            .make_audio_decoder(audio_params, &decoder_opts)?;

        let total_duration = time_base.zip(total_ticks).and_then(|(base, ticks)| {
            base.calc_time(symphonia::core::units::Timestamp::from(ticks as i64))
                .map(time_to_std_duration)
                .filter(|d| !d.is_zero())
        });

        let mut decoded_buffer = Vec::<Sample>::new();
        let (spec, has_decoded) = loop {
            let packet = match format.next_packet() {
                Ok(Some(packet)) => packet,
                Ok(None) => {
                    let last = decoder.last_decoded();
                    if last.frames() > 0 {
                        let spec = last.spec().clone();
                        last.copy_to_vec_interleaved(&mut decoded_buffer);
                        break (spec, true);
                    }
                    break (AudioSpec::default(), false);
                }
                Err(e) => return Err(e),
            };

            // If the packet does not belong to the selected track, skip over it
            if packet.track_id != track_id {
                continue;
            }

            match decoder.decode(&packet) {
                Ok(decoded) if decoded.frames() > 0 => {
                    let spec = decoded.spec().clone();
                    decoded.copy_to_vec_interleaved(&mut decoded_buffer);
                    break (spec, true);
                }
                Ok(_) => continue, // skip setup/header packets with no audio frames (e.g. Vorbis)
                Err(e) => match e {
                    Error::DecodeError(_) => {
                        // Decode errors are intentionally ignored with no retry limit.
                        // This behavior ensures that the decoder skips over problematic packets
                        // and continues processing the rest of the stream.
                        continue;
                    }
                    _ => return Err(e),
                },
            };
        };

        if !has_decoded {
            return Ok(None);
        }

        Ok(Some(SymphoniaDecoder {
            decoder,
            current_span_offset: 0,
            format,
            total_duration,
            buffer: decoded_buffer,
            spec,
            seek_mode,
            selected_track_id: track_id,
            time_base,
            samples_in_current_frame: 0,
            silence_samples_remaining: 0,
            seeked_past_end: false,
        }))
    }
}

#[inline]
fn time_to_std_duration(time: Time) -> Duration {
    let (secs, nanos) = time.parts();
    if secs >= 0 {
        Duration::new(secs as u64, nanos)
    } else {
        Duration::ZERO
    }
}

impl Source for SymphoniaDecoder {
    #[inline]
    fn current_span_len(&self) -> Option<usize> {
        Some(self.buffer.len())
    }

    #[inline]
    fn channels(&self) -> ChannelCount {
        ChannelCount::new(
            self.spec
                .channels()
                .count()
                .try_into()
                .expect("rodio only support up to u16::MAX channels (65_535)"),
        )
        .expect("audio should always have at least one channel")
    }

    #[inline]
    fn sample_rate(&self) -> SampleRate {
        SampleRate::new(self.spec.rate()).expect("audio should always have a non zero SampleRate")
    }

    #[inline]
    fn total_duration(&self) -> Option<Duration> {
        self.total_duration
    }

    fn try_seek(&mut self, pos: Duration) -> Result<(), source::SeekError> {
        if matches!(self.seek_mode, SeekMode::Accurate) && self.time_base.is_none() {
            return Err(source::SeekError::SymphoniaDecoder(
                SeekError::AccurateSeekNotSupported,
            ));
        }

        // Seeking should be "saturating", meaning: target positions beyond the end of the stream
        // are clamped to the end.
        let mut target = pos;
        if let Some(total_duration) = self.total_duration {
            if target > total_duration {
                target = total_duration;
            }
        }

        let target_time = Time::try_from_secs_f64(target.as_secs_f64())
            .ok_or(source::SeekError::SymphoniaDecoder(
                SeekError::AccurateSeekNotSupported,
            ))?;

        // Remember the current channel, so we can restore it after seeking.
        let active_channel = self.current_span_offset % self.channels().get() as usize;

        let seek_res = match self.format.seek(
            self.seek_mode,
            SeekTo::Time {
                time: target_time,
                track_id: None,
            },
        ) {
            Err(Error::SeekError(symphonia::core::errors::SeekErrorKind::ForwardOnly)) => {
                return Err(source::SeekError::SymphoniaDecoder(
                    SeekError::RandomAccessNotSupported,
                ));
            }
            Err(Error::SeekError(symphonia::core::errors::SeekErrorKind::OutOfRange)) => {
                // Saturate seeks past the end of the stream to the end.
                self.decoder.reset();
                self.current_span_offset = usize::MAX;
                self.samples_in_current_frame = 0;
                self.silence_samples_remaining = 0;
                self.seeked_past_end = true;
                return Ok(());
            }
            other => other.map_err(Arc::new).map_err(SeekError::Demuxer),
        }?;

        // Seeking is a demuxer operation without the decoder knowing about it,
        // so we need to reset the decoder to make sure it's in sync and prevent
        // audio glitches.
        self.decoder.reset();

        // Force the iterator to decode the next packet.
        self.current_span_offset = usize::MAX;
        self.seeked_past_end = false;

        // Symphonia does not seek to the exact position, it seeks to the closest keyframe.
        // If accurate seeking is required, fast-forward to the exact position.
        if matches!(self.seek_mode, SeekMode::Accurate) {
            self.refine_position(seek_res)?;
        }

        // After seeking, we are at the beginning of an inter-sample frame, i.e. the first
        // channel. We need to advance the iterator to the right channel.
        for _ in 0..active_channel {
            self.next();
        }

        Ok(())
    }
}

/// Error returned when the try_seek implementation of the symphonia decoder fails.
#[derive(Debug, thiserror::Error, Clone)]
pub enum SeekError {
    /// Accurate seeking is not supported
    ///
    /// This error occurs when the decoder cannot extract time base information from the source.
    /// You may catch this error to try a coarse seek instead.
    #[error("Accurate seeking is not supported on this file/byte stream that lacks time base information")]
    AccurateSeekNotSupported,
    /// The decoder does not support random access seeking
    ///
    /// This error occurs when the source is not seekable or does not have a known byte length.
    #[error("The decoder needs to know the length of the file/byte stream to be able to seek backwards. You can set that by using the `DecoderBuilder` or creating a decoder using `Decoder::try_from(some_file)`.")]
    RandomAccessNotSupported,
    /// Demuxer failed to seek
    #[error("Demuxer failed to seek")]
    Demuxer(#[source] Arc<symphonia::core::errors::Error>),
}
assert_error_traits!(SeekError);

impl SymphoniaDecoder {
    /// Note span offset must be set after
    fn refine_position(&mut self, seek_res: SeekedTo) -> Result<(), source::SeekError> {
        let time_base = self
            .time_base
            .expect("time base availability guaranteed by caller");

        // Calculate the time delta between requested and actual seek position.
        let delta_ticks = (seek_res.required_ts.get() - seek_res.actual_ts.get()).max(0);
        let delta = time_base
            .calc_time(symphonia::core::units::Timestamp::new(delta_ticks))
            .map(time_to_std_duration)
            .unwrap_or(Duration::ZERO);

        // Calculate the number of samples to skip.
        let mut samples_to_skip = (delta.as_secs_f32()
            * self.sample_rate().get() as f32
            * self.channels().get() as f32)
            .ceil() as usize;

        // Re-align the seek position to the first channel.
        samples_to_skip -= samples_to_skip % self.channels().get() as usize;

        // Skip ahead to the precise position.
        for _ in 0..samples_to_skip {
            self.next();
        }

        Ok(())
    }
}

impl Iterator for SymphoniaDecoder {
    type Item = Sample;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If padding to complete a frame, return silence
            if self.silence_samples_remaining > 0 {
                self.silence_samples_remaining -= 1;
                return Some(Sample::EQUILIBRIUM);
            }

            if self.seeked_past_end {
                return None;
            }

            if self.current_span_offset >= self.buffer.len() {
                let decoded_spec = loop {
                    let packet = match self.format.next_packet() {
                        Ok(Some(packet)) => {
                            if packet.track_id == self.selected_track_id {
                                packet
                            } else {
                                continue;
                            }
                        }
                        Ok(None) | Err(_) => {
                            // Input exhausted - check if mid-frame
                            let channels = self.channels();
                            self.silence_samples_remaining =
                                padding_samples_needed(self.samples_in_current_frame, channels);
                            if self.silence_samples_remaining > 0 {
                                self.samples_in_current_frame = 0;
                                break None;
                            }
                            return None;
                        }
                    };
                    let decoded = match self.decoder.decode(&packet) {
                        Ok(decoded) => decoded,
                        Err(Error::DecodeError(_)) => {
                            // Skip over packets that cannot be decoded. This ensures the iterator
                            // continues processing subsequent packets instead of terminating due to
                            // non-critical decode errors.
                            continue;
                        }
                        Err(_) => {
                            // Input exhausted - check if mid-frame
                            let channels = self.channels();
                            self.silence_samples_remaining =
                                padding_samples_needed(self.samples_in_current_frame, channels);
                            if self.silence_samples_remaining > 0 {
                                self.samples_in_current_frame = 0;
                                break None;
                            }
                            return None;
                        }
                    };

                    // Loop until we get a packet with audio frames. This is necessary because some
                    // formats can have packets with only metadata, particularly when rewinding, in
                    // which case the iterator would otherwise end with `None`.
                    if decoded.frames() > 0 {
                        let spec = decoded.spec().clone();
                        self.buffer.clear();
                        decoded.copy_to_vec_interleaved(&mut self.buffer);
                        break Some(spec);
                    }
                };

                match decoded_spec {
                    Some(spec) => {
                        self.spec = spec;
                        self.current_span_offset = 0;
                    }
                    None => {
                        // Break out happened due to exhaustion, continue to emit padding
                        continue;
                    }
                }
            }

            let sample = *self.buffer.get(self.current_span_offset)?;
            self.current_span_offset += 1;

            let channels = self.channels();
            self.samples_in_current_frame =
                (self.samples_in_current_frame + 1) % channels.get() as usize;

            return Some(sample);
        }
    }
}
