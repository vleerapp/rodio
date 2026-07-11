//! Convert audio between PCM formats.
//!
//! A PCM stream is described by three properties, each with a converter that changes it while
//! leaving the others intact:
//!
//! - **Sample type:** the numeric representation of each sample (e.g. `i16`, `f32`).
//!   Changed with [`SampleTypeConverter`].
//! - **Channel count:** the number of channels in the audio stream (e.g. mono, stereo, 5.1).
//!   Changed with [`ChannelCountConverter`], which duplicates or drops channels as needed.
//! - **Sample rate:** frames per second (e.g. 44.1 kHz to 48 kHz). Changed with
//!   [`SampleRateConverter`], which resamples the signal. See the [`sample_rate`] module for the
//!   available algorithms and quality presets.
//!
//! Each converter is a [`Source`](crate::Source) (and [`Iterator`]) adapter that wraps another
//! source, so they can be composed by nesting. To retarget a source to a fixed output format in
//! one step, prefer the higher-level [`UniformSourceIterator`](crate::source::UniformSourceIterator),
//! which applies channel-count and sample-rate conversion together in the correct order, or
//! [`Source::resample`](crate::Source::resample) for the sample rate alone.

pub use self::channels::ChannelCountConverter;
pub use self::sample::SampleTypeConverter;
pub use self::sample_rate::{
    Poly, PolyConfigBuilder, ResampleConfig, SampleRateConverter, Interpolation, SincConfigBuilder,
    WindowFunction,
};

mod channels;
mod sample;
pub mod sample_rate;
