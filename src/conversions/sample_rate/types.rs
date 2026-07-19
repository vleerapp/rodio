use crate::{ChannelCount, Float};
use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Sub, SubAssign};

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
    () => {
        #[allow(dead_code)]
        pub const ZERO: Self = Self(0);
        #[allow(dead_code)]
        pub const MAX: Self = Self(usize::MAX);

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
            num_wrapper_shared! {}
        }
        forward_math! {$name}
    };
}

sample_wrapper!(InSamples, InFrameCount);
sample_wrapper!(OutSamples, OutFrameCount);

macro_rules! frame_wrapper {
    ($name:ident, $samples:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        pub struct $name(pub usize);

        #[allow(dead_code)]
        impl $name {
            pub fn samples(&self, num_channels: ChannelCount) -> $samples {
                $samples(self.0 * num_channels.get() as usize)
            }
            num_wrapper_shared! {}
        }
        forward_math! {$name}
    };
}

frame_wrapper!(InFrameCount, InSamples);
frame_wrapper!(OutFrameCount, OutSamples);

macro_rules! in_wrapper_shared {
    ($in:ident, $out:ident) => {
        impl $in {
            #[allow(dead_code)]
            pub fn resampled_by(&self, ratio: Float) -> $out {
                let raw = self.raw() as Float * ratio;
                let raw = raw.ceil() as usize;
                $out(raw)
            }
        }
    };
}

in_wrapper_shared!(InFrameCount, OutFrameCount);
in_wrapper_shared!(InSamples, OutSamples);
