//! Rubato resampler wrapper and implementations.

use dasp_sample::Sample as _;
use num_rational::Ratio;
use rubato::Resampler;

use crate::source::{ChannelCount, SampleRate, Source};
use crate::{Float, Sample};

use super::buffer::Buffer;
use super::builder::{Poly, Sinc, WindowFunction};

#[derive(thiserror::Error, Debug)]
#[error("Failed to create resampler")]
pub(super) struct ResamplerCreationError(#[from] rubato::ResamplerConstructionError);

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

    /// Number of valid samples in the current output chunk.
    pub fn output_len(&self) -> usize {
        self.output_buffer.len()
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

    pub fn next_sample(&mut self) -> Option<Sample> {
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

            // Fill input buffer, flushing with zeros if input is exhausted
            let needed_input = self.resampler.input_frames_next();
            let frames_before = self.input_frame_count;
            self.fill_input_buffer(needed_input, num_channels);

            // We can process with fewer frames than needed using partial_len when the input is
            // exhausted. If we don't have enough input and more is coming, wait.
            let made_progress = self.input_frame_count > frames_before;
            if self.input_frame_count < needed_input && !self.input_exhausted && made_progress {
                continue;
            }

            let actual_frames = self.input_frame_count;

            let indexing;
            let indexing_ref = if actual_frames < needed_input {
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
                // InterleavedSlice is a zero-cost abstraction - no heap allocation occurs here
                let input_adapter = audioadapter_buffers::direct::InterleavedSlice::new(
                    &self.input_buffer,
                    num_channels,
                    actual_frames,
                )
                .ok()?;

                let num_frames = self.output_buffer.capacity() / num_channels;
                let mut output_adapter = audioadapter_buffers::direct::InterleavedSlice::new_mut(
                    self.output_buffer.as_mut_slice(),
                    num_channels,
                    num_frames,
                )
                .ok()?;

                self.resampler
                    .process_into_buffer(&input_adapter, &mut output_adapter, indexing_ref)
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

            // Shift remaining input samples to beginning of buffer
            if actual_consumed < self.input_frame_count {
                let src_start = actual_consumed * num_channels;
                let src_end = self.input_frame_count * num_channels;
                self.input_buffer.copy_within(src_start..src_end, 0);
            }
            self.input_frame_count -= actual_consumed;

            self.output_buffer.reset(frames_out * num_channels);

            // Skip warmup delay samples
            if self.output_delay_remaining > 0 {
                let samples_to_skip = self.output_delay_remaining.min(self.output_buffer.len());
                self.output_buffer.skip(samples_to_skip);
                self.output_delay_remaining -= samples_to_skip;
            }

            // Cap output to cut off filter artifacts once input is exhausted
            if self.input_exhausted && self.expected_output_samples > 0 {
                let remaining = self
                    .expected_output_samples
                    .saturating_sub(self.total_output_samples);
                self.output_buffer.cap_to_remaining(remaining);
            }
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
        interpolation: Sinc,
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

        // Calculate the GCD-reduced ratio
        let ratio = Ratio::new(target_rate.get(), source_rate.get());
        let (_num, den) = ratio.into_raw();

        // Determine input chunk size - must be multiple of denominator
        let input_chunk_size = ((chunk_size / den as usize) + 1) * den as usize;

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
            resample_ratio,
        })
    }
}
