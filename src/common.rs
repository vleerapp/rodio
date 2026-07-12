use std::fmt::{Debug, Display};
use std::num::NonZero;
use std::ops::{Add, AddAssign, Sub, SubAssign};

use crate::math::nz;

/// Sample rate (a frame rate or samples per second per channel).
pub type SampleRate = NonZero<u32>;

/// The default sample rate used by rodio for generators and device sinks.
pub const DEFAULT_SAMPLE_RATE: SampleRate = nz!(48_000);

/// Number of channels in a stream. Can never be Zero
pub type ChannelCount = NonZero<u16>;

/// Number of bits per sample. Can never be zero.
pub type BitDepth = NonZero<u32>;

// NOTE on numeric precision:
//
// While `f32` is transparent for typical playback use cases, it does not guarantee preservation of
// full 24-bit source fidelity across arbitrary processing chains. Each floating-point operation
// rounds its result to `f32` precision (~24-bit significand). In DSP pipelines (filters, mixing,
// modulation), many operations are applied per sample and over time, so rounding noise accumulates
// and long-running state (e.g. oscillator phase) can drift.
//
// For use cases where numerical accuracy must be preserved through extended processing (recording,
// editing, analysis, long-running generators, or complex DSP graphs), enabling 64-bit processing
// reduces accumulated rounding error and drift.
//
// This mirrors common practice in professional audio software and DSP libraries, which often use
// 64-bit internal processing even when the final output is 16- or 24-bit.

/// Floating point type used for internal calculations. Can be configured to be
/// either `f32` (default) or `f64` using the `64bit` feature flag.
#[cfg(not(feature = "64bit"))]
pub type Float = f32;

/// Floating point type used for internal calculations. Can be configured to be
/// either `f32` (default) or `f64` using the `64bit` feature flag.
#[cfg(feature = "64bit")]
pub type Float = f64;

/// Represents value of a single sample.
/// Silence corresponds to the value `0.0`. The expected amplitude range is  -1.0...1.0.
/// Values below and above this range are clipped in conversion to other sample types.
/// Use conversion traits from [dasp_sample] crate or [crate::conversions::SampleTypeConverter]
/// to convert between sample types if necessary.
pub type Sample = Float;

/// Used to test at compile time that a struct/enum implements Send, Sync and
/// is 'static. These are common requirements for dynamic error management
/// libs like color-eyre and anyhow
///
/// # Examples
/// ```compile_fail
/// struct NotSend {
///   foo: Rc<String>,
/// }
///
/// assert_error_traits!(NotSend)
/// ```
///
/// ```compile_fail
/// struct NotSync {
///   foo: std::cell::RefCell<String>,
/// }
/// assert_error_traits!(NotSync)
/// ```
///
/// ```compile_fail
/// struct NotStatic<'a> {
///   foo: &'a str,
/// }
///
/// assert_error_traits!(NotStatic)
/// ```
macro_rules! assert_error_traits {
    ($to_test:path) => {
        const _: () = { $crate::common::use_required_traits::<$to_test>() };
    };
}

pub(crate) use assert_error_traits;
#[allow(dead_code)]
pub(crate) const fn use_required_traits<T: Send + Sync + 'static + Display + Debug + Clone>() {}

macro_rules! forward_math {
    ($name:ident) => {
        impl AddAssign<usize> for $name {
            fn add_assign(&mut self, rhs: usize) {
                self.0 += rhs
            }
        }

        impl AddAssign<Self> for $name {
            fn add_assign(&mut self, rhs: Self) {
                self.0 += rhs.0
            }
        }
        impl SubAssign<Self> for $name {
            fn sub_assign(&mut self, rhs: Self) {
                self.0 -= rhs.0
            }
        }

        impl Add<Self> for $name {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                Self(self.0 + rhs.0)
            }
        }
        impl Add<usize> for $name {
            type Output = Self;

            fn add(self, rhs: usize) -> Self::Output {
                Self(self.0 + rhs)
            }
        }
        impl Sub<Self> for $name {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self::Output {
                Self(self.0 - rhs.0)
            }
        }

        impl $name {
            #[allow(dead_code)]
            #[must_use]
            pub fn saturating_sub(&self, rhs: Self) -> Self {
                Self(self.0.saturating_sub(rhs.0))
            }
        }
    };
}

macro_rules! num_wrapper_shared {
    ($neutral:ident) => {
        #[allow(dead_code)]
        pub const ZERO: Self = Self(0);
        #[allow(dead_code)]
        pub const MAX: Self = Self(usize::MAX);

        #[allow(dead_code)]
        #[must_use]
        pub fn count(&self) -> $neutral {
            $neutral(self.0)
        }

        #[allow(dead_code)]
        #[must_use]
        pub fn raw(&self) -> usize {
            self.0
        }

        #[allow(dead_code)]
        pub fn raw_mut(&mut self) -> &mut usize {
            &mut self.0
        }
    };
}

macro_rules! sample_wrapper {
    ($name:ident, $frames:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        pub struct $name(pub usize);

        impl $name {
            #[allow(dead_code)]
            pub fn frames(&self, num_channels: ChannelCount) -> $frames {
                $frames(&self.0 / num_channels.get() as usize)
            }
            num_wrapper_shared! {SampleCount}
        }
        forward_math! {$name}
    };
}

sample_wrapper!(InSamples, InFrameCount);
sample_wrapper!(OutSamples, OutFrameCount);
sample_wrapper!(SampleCount, FrameCount);

macro_rules! frame_wrapper {
    ($name:ident, $samples:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        pub struct $name(pub usize);

        #[allow(dead_code)]
        impl $name {
            pub fn samples(&self, num_channels: ChannelCount) -> $samples {
                $samples(self.0 * num_channels.get() as usize)
            }
            num_wrapper_shared! {FrameCount}
        }
        forward_math! {$name}
    };
}

frame_wrapper!(InFrameCount, InSamples);
frame_wrapper!(OutFrameCount, OutSamples);
frame_wrapper!(FrameCount, SampleCount);

macro_rules! in_wrapper_shared {
    ($in:ident, $out:ident) => {
        impl $in {
            #[allow(dead_code)]
            pub fn resampled_by(&self, ratio: f32) -> $out {
                let raw = self.raw() as Float * ratio as Float;
                let raw = raw.ceil() as usize;
                $out(raw)
            }
        }
    };
}

in_wrapper_shared!(InFrameCount, OutFrameCount);
in_wrapper_shared!(InSamples, OutSamples);
