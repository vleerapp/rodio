//! Rubato resampler wrapper and implementations.

use dasp_sample::Sample as _;
use rubato::{audioadapter_buffers::direct::InterleavedSlice, Resampler};

use crate::common::{ChannelCount, FrameCount, InFrameCount, InSamples, OutFrameCount, SampleRate};
use crate::conversions::sample_rate::buffer::Output;
use crate::{Float, Sample, Source};

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

    pub input_buffer: Vec<Sample>,
    pub(crate) output: super::buffer::Output,

    /// The following are cached at construction for parameter-change detection.
    pub resample_ratio: Float,

    pub output_delay_remaining: OutFrameCount,
    pub pos_in_current_span: InSamples,

    pub in_resampler_state: OutFrameCount,
}

impl<I: Source, R: rubato::Resampler<Sample>> RubatoResample<I, R> {
    /// Calculate the number of output samples to skip for delay compensation.
    pub fn output_delay(resampler: &R) -> OutFrameCount {
        // Skip delay-1 frames to align the first output frame with input position 0.
        let delay_frames = resampler.output_delay();
        let delay_frames = delay_frames.saturating_sub(1);
        OutFrameCount(delay_frames)
    }

    pub fn span_length(&self) -> Option<usize> {
        if !self.output.is_empty() {
            // rest of the output buffer is rest of span
            Some(self.output.current_span_len())
        } else if self.input.is_exhausted() {
            Some(0)
        } else {
            // True span length unknown, return the smallest possible span since that is
            // always correct
            Some(self.input.channels().get() as usize)
        }
    }

    /// Whether the output buffer has unconsumed samples.
    pub fn output_has_samples(&self) -> bool {
        !self.output.is_empty()
    }

    pub fn reset(&mut self) {
        self.resampler.reset();
        self.output.reset();
        self.pos_in_current_span = InSamples::ZERO;
        self.output_delay_remaining = Self::output_delay(&self.resampler);
    }

    pub fn next_sample(&mut self) -> Option<Sample> {
        loop {
            if let Some(sample) = self.output.next() {
                return Some(sample);
            }

            // We could get so much output delay that multiple chunks are
            // all silence which is why there is a loop here.
            std::hint::cold_path();
            self.resample_chunk()?;
        }
    }

    // Extracted such that rustc has an easier time outlining this.
    //
    // This is complicated since we need to handle changing sample rates (spans)
    #[inline(never)]
    fn resample_chunk(&mut self) -> Option<()> {
        dbg!(self.input.is_exhausted(), &self.output);
        if self.input.is_exhausted() && self.resampler_empty() {
            return None;
        }

        let frames_in = self.fill_input_buffer(self.output.channels);
        self.in_resampler_state += frames_in.resampled_by(self.resample_ratio);
        self.pos_in_current_span += frames_in.samples(self.output.channels);

        let needed_by_resampler = InFrameCount(self.resampler.input_frames_next());
        let partial_len = (frames_in < needed_by_resampler).then_some(frames_in.raw());

        let indexing = Some(&rubato::Indexing {
            input_offset: 0,
            output_offset: 0,
            partial_len,
            active_channels_mask: None,
        });
        dbg!(frames_in, needed_by_resampler, partial_len);

        let input_adapter = InterleavedSlice::new(
            &self.input_buffer,
            self.output.channels.get().into(),
            needed_by_resampler.raw(),
        )
        .expect("We always set up the input_buffer correctly");

        let channels = self.output.channels.get().into();
        let capacity = self.output.capacity().raw();
        let mut output_adapter = InterleavedSlice::new_mut(self.output.reset(), channels, capacity)
            .expect("We always set up the input_buffer correctly");

        let (_, frames_out) = self
            .resampler
            .process_into_buffer(&input_adapter, &mut output_adapter, indexing)
            .map(|(r#in, out)| (InFrameCount(r#in), OutFrameCount(out)))
            .expect("We always set up the input_buffer correctly");
        dbg!(frames_out);

        if self.input.is_exhausted() && frames_out == OutFrameCount::ZERO {
            return None;
        }

        let delay_in_output = self
            .output_delay_remaining
            .min(self.in_resampler_state.min(frames_out));
        self.output_delay_remaining -= delay_in_output;

        self.output.set_end(self.in_resampler_state.min(frames_out));
        self.output.set_start(delay_in_output);

        dbg!(self.in_resampler_state);
        self.in_resampler_state -= self.output.len().frames(self.output.channels);
        dbg!(self.in_resampler_state);

        Some(())
    }

    fn fill_input_buffer(&mut self, num_channels: ChannelCount) -> InFrameCount {
        let needed_by_resampler = InFrameCount(self.resampler.input_frames_next());
        let current_span_length = self
            .input
            .current_span_len()
            .map(InSamples)
            .map(|s| s.frames(num_channels));
        let frames_to_take =
            needed_by_resampler.min(current_span_length.unwrap_or(InFrameCount::MAX));

        let mut samples_taken = InSamples::ZERO;
        for _ in 0..frames_to_take.samples(num_channels).raw() {
            if let Some(sample) = self.input.next() {
                self.input_buffer.push(sample);
                samples_taken += 1usize;
            } else {
                break;
            }
        }
        samples_taken.frames(num_channels)
    }

    fn resampler_empty(&self) -> bool {
        self.in_resampler_state == OutFrameCount::ZERO
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
        let output_buf_size = FrameCount(resampler.output_frames_max());

        let initial_output_delay =
            RubatoResample::<I, rubato::Async<Sample>>::output_delay(&resampler);

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
        let output_buf_size = FrameCount(resampler.output_frames_max());

        let initial_output_delay =
            RubatoResample::<I, rubato::Async<Sample>>::output_delay(&resampler);

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
        output_buf_size: FrameCount,
        channels: ChannelCount,
        source_rate: SampleRate,
        initial_output_delay: OutFrameCount,
        resample_ratio: Float,
    ) -> Self {
        Self {
            input,
            resampler,
            input_buffer: vec![Sample::EQUILIBRIUM; input_buf_size.samples(channels).raw()],
            output: Output::new(source_rate, channels, output_buf_size),
            pos_in_current_span: InSamples::ZERO,
            output_delay_remaining: initial_output_delay,
            resample_ratio,
            in_resampler_state: OutFrameCount::ZERO,
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
            output_buffer: OutputBuffer::new(output_buf_size * channels.get() as usize),
            channels,
            source_rate,
            input_samples_consumed: 0,
            input_exhausted: false,
            total_input_frames: 0,
            total_output_samples: 0,
            real_frames_in_buffer: 0,
            output_delay_remaining,
            output_span_len: 0,
            resample_ratio,
        })
    }
}
