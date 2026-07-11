//! Rubato resampler wrapper and implementations.

use dasp_sample::Sample as _;
use rubato::{audioadapter_buffers::direct::InterleavedSlice, Resampler};

use crate::common::{ChannelCount, SampleRate};
use crate::{Float, Sample, Source};

use super::buffer::Buffer;
use super::builder::{Poly, Interpolation, WindowFunction};

#[derive(thiserror::Error, Debug)]
#[error("Failed to create resampler")]
pub struct ResamplerCreationError(#[from] rubato::ResamplerConstructionError);

/// Type alias for Async (polynomial/sinc) resampler.
pub type RubatoAsyncResample<I> = RubatoResample<I, rubato::Async<Sample>>;

/// Type alias for FFT resampler (synchronous, fixed-ratio).
#[cfg(feature = "rubato-fft")]
pub type RubatoFftResample<I> = RubatoResample<I, rubato::Fft<Sample>>;

/// The inner resampler implementation chosen based on configuration and sample rates.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum ResampleInner<I: Source> {
    /// Passthrough when source rate is equal to the target rate
    Passthrough {
        source: I,
        input_span_pos: usize,
        channels: ChannelCount,
        source_rate: SampleRate,
    },

    /// Polynomial resampling (fast, no anti-aliasing)
    Poly(RubatoAsyncResample<I>),

    /// Sinc resampling (with anti-aliasing)
    Sinc(RubatoAsyncResample<I>),

    /// FFT resampling for fixed ratios (synchronous resampling)
    #[cfg(feature = "rubato-fft")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rubato-fft")))]
    Fft(RubatoFftResample<I>),
}

impl<I: Source> ResampleInner<I> {
    /// Get a reference to the inner input source
    #[inline]
    pub fn input(&self) -> &I {
        match self {
            ResampleInner::Passthrough { source, .. } => source,
            ResampleInner::Poly(resampler) => &resampler.input,
            ResampleInner::Sinc(resampler) => &resampler.input,
            #[cfg(feature = "rubato-fft")]
            ResampleInner::Fft(resampler) => &resampler.input,
        }
    }

    /// Extract the inner input source, consuming the resampler
    #[inline]
    pub fn into_inner(self) -> I {
        match self {
            ResampleInner::Passthrough { source, .. } => source,
            ResampleInner::Poly(resampler) => resampler.input,
            ResampleInner::Sinc(resampler) => resampler.input,
            #[cfg(feature = "rubato-fft")]
            ResampleInner::Fft(resampler) => resampler.input,
        }
    }
}

/// Generic wrapper around Rubato resamplers for sample-by-sample iteration.
#[derive(Debug)]
pub struct RubatoResample<I: Source, R: rubato::Resampler<Sample>> {
    pub input: I,
    pub resampler: R,

    pub input_buffer: Box<[Sample]>,
    pub input_frame_count: usize,

    output_buffer: Buffer,

    /// The following are cached at construction for parameter-change detection.
    pub channels: ChannelCount,
    pub source_rate: SampleRate,

    pub input_samples_consumed: usize,
    pub input_exhausted: bool,

    pub total_input_frames: usize,
    pub total_output_samples: usize,
    pub expected_output_samples: usize,

    /// The number of real (non-flush) frames currently in the input buffer.
    pub real_frames_in_buffer: usize,

    pub output_delay_remaining: usize,
    pub resample_ratio: Float,

    /// Effective length of the current output chunk as seen by callers of `current_span_len`.
    /// Differs from `output_buffer.len()` when delay-compensation skip consumed leading samples.
    pub output_span_len: usize,
}

impl<I: Source, R: rubato::Resampler<Sample>> RubatoResample<I, R> {
    /// Calculate the number of output samples to skip for delay compensation.
    pub fn calculate_delay_compensation(resampler: &R, channels: ChannelCount) -> usize {
        // Skip delay-1 frames to align the first output frame with input position 0.
        let delay_frames = resampler.output_delay();
        let delay_to_skip = delay_frames.saturating_sub(1);
        delay_to_skip * channels.get() as usize
    }

    /// Whether the output buffer has unconsumed samples.
    pub fn output_has_samples(&self) -> bool {
        !self.output_buffer.is_empty()
    }

    /// Effective span length of the current output chunk for `current_span_len` reporting.
    pub fn output_span_len(&self) -> usize {
        self.output_span_len
    }

    /// Number of output samples remaining to be read.
    pub fn output_remaining(&self) -> usize {
        self.output_buffer.remaining()
    }

    pub fn reset(&mut self) {
        self.resampler.reset();
        self.output_buffer.reset(0);
        self.input_frame_count = 0;
        self.input_samples_consumed = 0;
        self.input_exhausted = false;
        self.total_input_frames = 0;
        self.total_output_samples = 0;
        self.expected_output_samples = 0;
        self.real_frames_in_buffer = 0;
        self.output_delay_remaining =
            Self::calculate_delay_compensation(&self.resampler, self.channels);
        self.output_span_len = 0;
    }

    fn fill_input_buffer(&mut self, needed: usize, num_channels: usize) {
        while self.input_frame_count < needed {
            if self.input_exhausted {
                break;
            }
            let sample_pos = self.input_frame_count * num_channels;
            for ch in 0..num_channels {
                if let Some(sample) = self.input.next() {
                    self.input_buffer[sample_pos + ch] = sample;
                } else {
                    self.input_exhausted = true;
                    break;
                }
            }
            if !self.input_exhausted {
                self.input_frame_count += 1;
                self.real_frames_in_buffer += 1;
            }
        }

        // Zero-pad if we ran out of input to flush the filter tail
        if self.input_frame_count == 0 {
            self.input_buffer[..needed * num_channels].fill(Sample::EQUILIBRIUM);
            self.input_frame_count = needed;
            // real_frames_in_buffer stays at 0 - these are flush frames
        }
    }

    pub fn next_sample(&mut self, cached_input_span_len: Option<usize>) -> Option<Sample> {
        let num_channels = self.channels.get() as usize;
        loop {
            // If we have buffered output, return it
            if !self.output_buffer.is_empty() {
                let sample = self.output_buffer.read();
                self.total_output_samples += 1;
                return Some(sample);
            }

            // Need more input - first check if we're completely done
            if self.input_exhausted
                && self.input_frame_count == 0
                && self.total_output_samples >= self.expected_output_samples
            {
                return None;
            }

            // Fill input buffer, flushing with zeros if input is exhausted.
            // Cap to the span boundary so we never read into the next span's samples,
            // which may have a different channel count or sample rate.
            let original_needed = self.resampler.input_frames_next();
            let needed_input = if let Some(span_len) = cached_input_span_len {
                let span_frames = span_len / num_channels;
                // input_samples_consumed resets to 0 at each span boundary, so this
                // stays span-relative even when the resampler processes multiple spans.
                let already_read =
                    self.input_samples_consumed / num_channels + self.real_frames_in_buffer;
                let remaining = span_frames.saturating_sub(already_read);
                original_needed.min(self.input_frame_count + remaining)
            } else {
                original_needed
            };

            // When the span cap brings needed_input to zero and the buffer is empty,
            // check whether the source is truly exhausted (single-span done) or just
            // at a same-format span boundary. For the latter, SampleRateConverter::next()
            // will have refreshed cached_input_span_len before the next call.
            if needed_input == 0 && !self.input_exhausted {
                self.input_exhausted = self.input.is_exhausted();
            }

            // When exhausted, use a full chunk so zero-padding flushes the filter tail.
            let fill_target = if self.input_exhausted {
                original_needed
            } else {
                needed_input
            };
            self.fill_input_buffer(fill_target, num_channels);

            let actual_frames = self.input_frame_count;

            // Use original_needed (pre-cap) so that partial_len is signalled to Rubato
            // whenever we have fewer frames than a full chunk, regardless of whether the
            // shortfall is due to source exhaustion or a span boundary cap.
            let indexing;
            let indexing_ref = if actual_frames < original_needed {
                indexing = rubato::Indexing {
                    input_offset: 0,
                    output_offset: 0,
                    partial_len: Some(actual_frames),
                    active_channels_mask: None,
                };
                Some(&indexing)
            } else {
                None
            };

            let (frames_in, frames_out) = {
                let input_adapter =
                    InterleavedSlice::new(&self.input_buffer, num_channels, actual_frames)
                        .inspect_err(|_e| {
                            #[cfg(feature = "tracing")]
                            tracing::error!("resampler: failed to create input adapter: {_e}");
                        })
                        .ok()?;

                let num_frames = self.output_buffer.capacity() / num_channels;
                let mut output_adapter = InterleavedSlice::new_mut(
                    self.output_buffer.as_mut_slice(),
                    num_channels,
                    num_frames,
                )
                .inspect_err(|_e| {
                    #[cfg(feature = "tracing")]
                    tracing::error!("resampler: failed to create output adapter: {_e}");
                })
                .ok()?;

                self.resampler
                    .process_into_buffer(&input_adapter, &mut output_adapter, indexing_ref)
                    .inspect_err(|_e| {
                        #[cfg(feature = "tracing")]
                        tracing::error!("resampler: processing failed: {_e}");
                    })
                    .ok()?
            };

            // If no output was produced and input is exhausted, we're done
            if frames_out == 0 && self.input_exhausted {
                return None;
            }

            // When using partial_len, Rubato may report consuming more frames than we
            // actually provided (it counts the zero-padded frames). Clamp to actual.
            let actual_consumed = frames_in.min(actual_frames);
            self.input_samples_consumed += actual_consumed * num_channels;

            // Only count real (non-flush) frames toward expected output
            let real_consumed = actual_consumed.min(self.real_frames_in_buffer);
            self.real_frames_in_buffer -= real_consumed;
            self.total_input_frames += real_consumed;
            self.expected_output_samples = (self.total_input_frames as Float * self.resample_ratio)
                .ceil() as usize
                * num_channels;

            self.input_frame_count -= actual_consumed;

            self.output_buffer.reset(frames_out * num_channels);

            // Skip warmup delay samples
            if self.output_delay_remaining > 0 {
                let samples_to_skip = self.output_delay_remaining.min(self.output_buffer.len());
                self.output_buffer.skip(samples_to_skip);
                self.output_delay_remaining -= samples_to_skip;
            }

            // Cap output whenever a partial chunk was processed (span boundary or
            // exhaustion) or once all input is gone. Rubato internally zero-pads a
            // partial chunk to a full output chunk, so without the cap the output
            // would exceed expected_output_samples.
            if (self.input_exhausted || indexing_ref.is_some()) && self.expected_output_samples > 0
            {
                let remaining = self
                    .expected_output_samples
                    .saturating_sub(self.total_output_samples);
                self.output_buffer.cap_to_remaining(remaining);
            }

            // Snapshot remaining after skip and cap. Stays constant while the chunk drains,
            // giving current_span_len a stable total that excludes delay-skipped leading samples.
            self.output_span_len = self.output_buffer.remaining();
        }
    }
}

// Async resampler (polynomial and sinc) implementations
impl<I: Source> RubatoAsyncResample<I> {
    pub fn new_poly(
        input: I,
        target_rate: SampleRate,
        chunk_size: usize,
        degree: Poly,
    ) -> Result<Self, ResamplerCreationError> {
        let source_rate = input.sample_rate();
        let channels = input.channels();

        let resample_ratio = target_rate.get() as Float / source_rate.get() as Float;

        let resampler = rubato::Async::new_poly(
            resample_ratio as _,
            1.0,
            degree.into(),
            chunk_size,
            channels.get() as usize,
            rubato::FixedAsync::Output,
        )?;

        let input_buf_size = resampler.input_frames_max();
        let output_buf_size = resampler.output_frames_max();

        let output_delay_remaining =
            RubatoResample::<I, rubato::Async<Sample>>::calculate_delay_compensation(
                &resampler, channels,
            );

        Ok(Self {
            input,
            resampler,
            input_buffer: vec![Sample::EQUILIBRIUM; input_buf_size * channels.get() as usize]
                .into_boxed_slice(),
            input_frame_count: 0,
            output_buffer: Buffer::new(output_buf_size * channels.get() as usize),
            channels,
            source_rate,
            input_samples_consumed: 0,
            input_exhausted: false,
            output_delay_remaining,
            output_span_len: 0,
            total_input_frames: 0,
            total_output_samples: 0,
            expected_output_samples: 0,
            real_frames_in_buffer: 0,
            resample_ratio,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_sinc(
        input: I,
        target_rate: SampleRate,
        chunk_size: usize,
        sinc_len: usize,
        f_cutoff: Float,
        oversampling_factor: usize,
        interpolation: Interpolation,
        window: WindowFunction,
    ) -> Result<Self, ResamplerCreationError> {
        let source_rate = input.sample_rate();
        let channels = input.channels();

        let parameters = rubato::SincInterpolationParameters {
            sinc_len,
            f_cutoff: f_cutoff as _,
            oversampling_factor,
            interpolation: interpolation.into(),
            window: window.into(),
        };

        let resample_ratio = target_rate.get() as Float / source_rate.get() as Float;

        let resampler = rubato::Async::new_sinc(
            resample_ratio as _,
            1.0,
            &parameters,
            chunk_size,
            channels.get() as usize,
            rubato::FixedAsync::Output,
        )?;

        let input_buf_size = resampler.input_frames_max();
        let output_buf_size = resampler.output_frames_max();

        let output_delay_remaining =
            RubatoResample::<I, rubato::Async<Sample>>::calculate_delay_compensation(
                &resampler, channels,
            );

        Ok(Self {
            input,
            resampler,
            input_buffer: vec![Sample::EQUILIBRIUM; input_buf_size * channels.get() as usize]
                .into_boxed_slice(),
            input_frame_count: 0,
            output_buffer: Buffer::new(output_buf_size * channels.get() as usize),
            channels,
            source_rate,
            input_samples_consumed: 0,
            input_exhausted: false,
            output_delay_remaining,
            output_span_len: 0,
            total_input_frames: 0,
            total_output_samples: 0,
            expected_output_samples: 0,
            real_frames_in_buffer: 0,
            resample_ratio,
        })
    }
}

// FFT resampler implementation
#[cfg(feature = "rubato-fft")]
impl<I: Source> RubatoFftResample<I> {
    /// Create a new FFT resampler for fixed-ratio sample rate conversion.
    ///
    /// The FFT resampler requires that:
    /// - Input chunk size must be a multiple of the GCD-reduced denominator
    /// - Output chunk size must be a multiple of the GCD-reduced numerator
    pub fn new(
        input: I,
        target_rate: SampleRate,
        chunk_size: usize,
        sub_chunks: usize,
    ) -> Result<Self, ResamplerCreationError> {
        let source_rate = input.sample_rate();
        let channels = input.channels();

        // Determine input chunk size - must be multiple of the GCD-reduced denominator
        let g = crate::math::gcd(target_rate.get(), source_rate.get());
        let den = (source_rate.get() / g) as usize;
        let input_chunk_size = ((chunk_size / den) + 1) * den;

        let resampler = rubato::Fft::new(
            source_rate.get() as usize,
            target_rate.get() as usize,
            input_chunk_size,
            sub_chunks,
            channels.get() as usize,
            rubato::FixedSync::Output,
        )?;

        let input_buf_size = resampler.input_frames_max();
        let output_buf_size = resampler.output_frames_max();
        let resample_ratio = target_rate.get() as Float / source_rate.get() as Float;

        let output_delay_remaining = Self::calculate_delay_compensation(&resampler, channels);

        Ok(Self {
            input,
            resampler,
            input_buffer: vec![Sample::EQUILIBRIUM; input_buf_size * channels.get() as usize]
                .into_boxed_slice(),
            input_frame_count: 0,
            output_buffer: Buffer::new(output_buf_size * channels.get() as usize),
            channels,
            source_rate,
            input_samples_consumed: 0,
            input_exhausted: false,
            total_input_frames: 0,
            total_output_samples: 0,
            expected_output_samples: 0,
            real_frames_in_buffer: 0,
            output_delay_remaining,
            output_span_len: 0,
            resample_ratio,
        })
    }
}
