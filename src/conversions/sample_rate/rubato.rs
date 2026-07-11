//! Rubato resampler wrapper and implementations.

use dasp_sample::Sample as _;
use rubato::{audioadapter_buffers::direct::InterleavedSlice, Resampler};

use crate::common::{ChannelCount, InFrameCount, InSamples, OutFrameCount, OutSamples, SampleRate};
use crate::{Float, Sample, Source};

use super::buffer::OutputBuffer;
use super::builder::{Interpolation, Poly, WindowFunction};

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
        input_span_pos: InSamples,
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
    pub input_frame_count: InFrameCount,

    output_buffer: OutputBuffer,

    /// The following are cached at construction for parameter-change detection.
    pub channels: ChannelCount,
    pub source_rate: SampleRate,

    pub input_samples_consumed: InSamples,
    pub input_exhausted: bool,

    pub total_input_frames: InFrameCount,
    pub total_output_samples: OutSamples,
    pub expected_output_samples: OutSamples,

    /// The number of real (non-flush) frames currently in the input buffer.
    pub real_frames_in_buffer: InFrameCount,

    pub output_delay_remaining: OutSamples,
    pub resample_ratio: Float,

    /// Effective length of the current output chunk as seen by callers of `current_span_len`.
    /// Differs from `output_buffer.len()` when delay-compensation skip consumed leading samples.
    pub output_span_len: OutSamples,
}

impl<I: Source, R: rubato::Resampler<Sample>> RubatoResample<I, R> {
    /// Calculate the number of output samples to skip for delay compensation.
    pub fn calculate_delay_compensation(resampler: &R, channels: ChannelCount) -> OutSamples {
        // Skip delay-1 frames to align the first output frame with input position 0.
        let delay_frames = resampler.output_delay();
        let delay_frames = delay_frames.saturating_sub(1);
        OutFrameCount(delay_frames).samples(channels)
    }

    /// Whether the output buffer has unconsumed samples.
    pub fn output_has_samples(&self) -> bool {
        !self.output_buffer.is_empty()
    }

    /// Effective span length of the current output chunk for `current_span_len` reporting.
    pub fn output_span_len(&self) -> OutSamples {
        self.output_span_len
    }

    /// Number of output samples remaining to be read.
    pub fn output_remaining(&self) -> OutSamples {
        self.output_buffer.remaining()
    }

    pub fn reset(&mut self) {
        self.resampler.reset();
        self.output_buffer.rewind_to(OutSamples::ZERO);
        self.input_frame_count = InFrameCount::ZERO;
        self.input_samples_consumed = InSamples::ZERO;
        self.input_exhausted = false;
        self.total_input_frames = InFrameCount::ZERO;
        self.total_output_samples = OutSamples::ZERO;
        self.expected_output_samples = OutSamples::ZERO;
        self.real_frames_in_buffer = InFrameCount::ZERO;
        self.output_delay_remaining =
            Self::calculate_delay_compensation(&self.resampler, self.channels);
        self.output_span_len = OutSamples::ZERO;
    }

    fn extend_buffer_from_source(&mut self, needed: InFrameCount, num_channels: ChannelCount) {
        'outer: while self.input_frame_count.count() < needed.count() {
            let frame = self.input_frame_count.samples(num_channels);
            let next = (self.input_frame_count + 1).samples(num_channels);
            for i in frame.raw()..next.raw() {
                if let Some(sample) = self.input.next() {
                    self.input_buffer[i] = sample;
                } else {
                    self.input_exhausted = true;
                    break 'outer;
                }
            }

            self.input_frame_count += 1;
            self.real_frames_in_buffer += 1;
        }

        // Zero-pad if we ran out of input to flush the filter tail
        if self.input_frame_count == InFrameCount::ZERO {
            self.input_buffer[..needed.samples(num_channels).raw()].fill(Sample::EQUILIBRIUM);
            self.input_frame_count = needed;
            // real_frames_in_buffer stays at 0 - these are flush frames
        }
    }

    pub fn next_sample(&mut self, cached_input_span_len: Option<InSamples>) -> Option<Sample> {
        loop {
            if !self.output_buffer.is_empty() {
                let sample = self.output_buffer.read();
                self.total_output_samples += 1;
                return Some(sample);
            }

            std::hint::cold_path();
            self.resample_chunk(cached_input_span_len, self.channels)?;
        }
    }

    // Extracted such that rustc has an easier time outlining this.
    //
    // This is complicated since we need to handle changing sample rates (spans)
    #[inline(never)]
    fn resample_chunk(
        &mut self,
        cached_input_span_len: Option<InSamples>,
        num_channels: ChannelCount,
    ) -> Option<()> {
        if self.input_exhausted
            && self.input_frame_count == InFrameCount::ZERO
            && self.total_output_samples >= self.expected_output_samples
        {
            return None;
        }
        let original_needed = self.fill_input_buffer(cached_input_span_len, num_channels);

        let indexing = if self.input_frame_count < original_needed {
            Some(&rubato::Indexing {
                input_offset: 0,
                output_offset: 0,
                partial_len: Some(self.input_frame_count.raw()),
                active_channels_mask: None,
            })
        } else {
            None
        };

        let input_adapter = InterleavedSlice::new(
            &self.input_buffer,
            num_channels.get().into(),
            self.input_frame_count.raw(),
        )
        .expect("We always set up the input_buffer correctly");

        let num_frames = self.output_buffer.capacity().frames(num_channels);
        let mut output_adapter = InterleavedSlice::new_mut(
            self.output_buffer.as_mut_slice(),
            num_channels.get().into(),
            num_frames.raw(),
        )
        .expect("We always set up the input_buffer correctly");

        let (frames_in, frames_out) = self
            .resampler
            .process_into_buffer(&input_adapter, &mut output_adapter, indexing)
            .map(|(r#in, out)| (InFrameCount(r#in), OutFrameCount(out)))
            .expect("We always set up the input_buffer correctly");

        if frames_out == OutFrameCount::ZERO && self.input_exhausted {
            return None;
        }

        self.update_output_buffer_cursor(num_channels, indexing, frames_in, frames_out);
        Some(())
    }

    fn update_output_buffer_cursor(
        &mut self,
        num_channels: std::num::NonZero<u16>,
        indexing: Option<&rubato::Indexing>,
        frames_in: InFrameCount,
        frames_out: OutFrameCount,
    ) {
        let frames_without_padding = frames_in.min(self.input_frame_count);
        self.input_samples_consumed += frames_without_padding.samples(num_channels);
        let real_consumed = frames_without_padding.min(self.real_frames_in_buffer);
        self.real_frames_in_buffer -= real_consumed;
        self.total_input_frames += real_consumed;
        self.input_frame_count -= frames_without_padding;
        self.output_buffer
            .rewind_to(frames_out.samples(num_channels));

        if self.output_delay_remaining > OutSamples::ZERO {
            let skipped = self.output_buffer.skip(self.output_delay_remaining);
            self.output_delay_remaining -= skipped;
        }

        if (self.input_exhausted || indexing.is_some())
            && self.expected_output_samples(num_channels) > OutSamples::ZERO
        {
            let remaining = self
                .expected_output_samples(num_channels)
                .saturating_sub(self.total_output_samples);
            self.output_buffer.cap_to_remaining(remaining);
        }
        self.output_span_len = self.output_buffer.remaining();
    }

    fn expected_output_samples(&self, num_channels: ChannelCount) -> OutSamples {
        let frames = self.total_input_frames.raw() as Float * self.resample_ratio;
        OutFrameCount(frames.ceil() as usize).samples(num_channels)
    }

    fn fill_input_buffer(
        &mut self,
        cached_input_span_len: Option<InSamples>,
        num_channels: ChannelCount,
    ) -> InFrameCount {
        let original_needed = InFrameCount(self.resampler.input_frames_next());
        let needed_input = if let Some(span_len) = cached_input_span_len {
            // input_samples_consumed resets to 0 at each span boundary, so this
            // stays span-relative even when the resampler processes multiple spans.
            let already_read =
                self.input_samples_consumed.frames(num_channels) + self.real_frames_in_buffer;
            let remaining = span_len.frames(num_channels).saturating_sub(already_read);
            original_needed.min(self.input_frame_count + remaining)
        } else {
            original_needed
        };

        if needed_input == InFrameCount::ZERO && !self.input_exhausted {
            self.input_exhausted = self.input.is_exhausted();
        }
        if self.input_exhausted {
            // When exhausted, use a full chunk so zero-padding flushes the filter tail.
            self.extend_buffer_from_source(original_needed, num_channels);
        } else {
            self.extend_buffer_from_source(needed_input, num_channels);
        };
        original_needed
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

        let input_buf_size = InFrameCount(resampler.input_frames_max());
        let output_buf_size = OutFrameCount(resampler.output_frames_max());

        let initial_output_delay =
            RubatoResample::<I, rubato::Async<Sample>>::calculate_delay_compensation(
                &resampler, channels,
            );

        Ok(Self::new_from(
            input,
            resampler,
            input_buf_size,
            output_buf_size,
            channels,
            source_rate,
            initial_output_delay,
            resample_ratio,
        ))
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

        let input_buf_size = InFrameCount(resampler.input_frames_max());
        let output_buf_size = OutFrameCount(resampler.output_frames_max());

        let initial_output_delay =
            RubatoResample::<I, rubato::Async<Sample>>::calculate_delay_compensation(
                &resampler, channels,
            );

        Ok(Self::new_from(
            input,
            resampler,
            input_buf_size,
            output_buf_size,
            channels,
            source_rate,
            initial_output_delay,
            resample_ratio,
        ))
    }

    fn new_from(
        input: I,
        resampler: rubato::Async<Sample>,
        input_buf_size: InFrameCount,
        output_buf_size: OutFrameCount,
        channels: ChannelCount,
        source_rate: SampleRate,
        initial_output_delay: OutSamples,
        resample_ratio: Float,
    ) -> Self {
        Self {
            input,
            resampler,
            input_buffer: vec![Sample::EQUILIBRIUM; input_buf_size.samples(channels).raw()]
                .into_boxed_slice(),
            input_frame_count: InFrameCount::ZERO,
            output_buffer: OutputBuffer::new(output_buf_size.samples(channels)),
            channels,
            source_rate,
            input_samples_consumed: InSamples::ZERO,
            input_exhausted: false,
            output_delay_remaining: initial_output_delay,
            output_span_len: OutSamples::ZERO,
            total_input_frames: InFrameCount::ZERO,
            total_output_samples: OutSamples::ZERO,
            expected_output_samples: OutSamples::ZERO,
            real_frames_in_buffer: InFrameCount::ZERO,
            resample_ratio,
        }
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
            output_buffer: OutputBuffer::new(output_buf_size * channels.get() as usize),
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
