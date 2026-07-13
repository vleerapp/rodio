//! Fixed-capacity sample buffer with a read cursor.
//!
use std::fmt::{Debug, Write};

use crate::common::{FrameCount, InSamples, OutFrameCount, OutSamples, SampleCount};
use crate::{ChannelCount, Sample, SampleRate};

pub(crate) struct Input {
    pub samples: Box<[Sample]>,
    pos: InSamples,
}

impl Debug for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Input")
            .field("pos", &self.pos)
            .field("data", &LimitLength(&self.samples[0..self.pos.raw()]))
            .finish()
    }
}

impl Input {
    pub(crate) fn new(capacity: InSamples) -> Self {
        let mut samples = Vec::new();
        samples.reserve_exact(capacity.raw());
        samples.resize(samples.capacity(), 0.0);
        Self {
            samples: samples.into_boxed_slice(),
            pos: InSamples::ZERO,
        }
    }

    pub(crate) fn push(&mut self, sample: Sample) {
        assert!(
            self.pos.raw() < self.samples.len(),
            "pos: {:?}, capacity: {}",
            self.pos,
            self.samples.len()
        );
        self.samples[self.pos.raw()] = sample;
        self.pos += 1;
    }

    pub(crate) fn as_slice(&mut self) -> &[Sample] {
        &self.samples
    }

    pub(crate) fn clear(&mut self) {
        self.pos = InSamples::ZERO;
    }

    pub(crate) fn len(&self) -> InSamples {
        self.pos
    }
}

pub(crate) struct Output {
    start: OutSamples,
    pos: OutSamples,
    end: OutSamples,

    pub samples: Box<[Sample]>,
    pub channels: ChannelCount,
    pub source_rate: SampleRate,
}

impl Debug for Output {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Output")
            .field("start", &self.start)
            .field("pos", &self.pos)
            .field("end", &self.end)
            .field(
                "data",
                &LimitLength(&self.samples[self.pos.raw()..self.end.raw()]),
            )
            .field("channels", &self.channels)
            .field("source_rate", &self.source_rate)
            .finish()
    }
}

struct LimitLength<'a>(&'a [Sample]);

impl Debug for LimitLength<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_empty() {
            return f.write_str("[]");
        } else if self.0.len() < 8 {
            return f.debug_list().entries(self.0).finish();
        }

        f.write_str("[")?;
        for element in self.0.iter().take(3) {
            f.write_str("\n\t")?;
            f.write_fmt(format_args!("{element:?}"))?;
            f.write_char(',')?;
        }

        if let Some(hidden) = self.0.len().checked_sub(6) {
            f.write_str("\n\t.. ")?;
            f.write_fmt(format_args!(" (hiding {hidden} entries)"))?;
        }

        for element in self.0.iter().rev().take(3).rev() {
            f.write_str("\n\t")?;
            f.write_fmt(format_args!("{element:?}"))?;
            f.write_char(',')?;
        }

        f.write_str("\n]")
    }
}

impl Output {
    pub(super) fn new(
        source_rate: SampleRate,
        channels: ChannelCount,
        capacity: FrameCount,
    ) -> Self {
        let mut samples = Vec::new();
        samples.reserve_exact(capacity.samples(channels).raw());
        samples.resize(samples.capacity(), 0.0);
        Self {
            start: OutSamples::ZERO,
            pos: OutSamples::ZERO,
            end: OutSamples::ZERO,
            samples: samples.into_boxed_slice(),
            channels,
            source_rate,
        }
    }

    pub(super) fn capacity(&self) -> FrameCount {
        SampleCount(self.samples.len()).frames(self.channels)
    }

    pub(super) fn len(&self) -> OutSamples {
        self.end - self.pos
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len().raw() == 0
    }

    pub(super) fn reset(&mut self) -> &mut [Sample] {
        self.pos = OutSamples::ZERO;
        self.start = OutSamples::ZERO;
        self.end = OutSamples::ZERO;
        &mut self.samples
    }

    pub(super) fn set_start(&mut self, start: OutFrameCount) {
        self.start = start.samples(self.channels);
        self.pos = self.start;
        assert!(self.start.raw() <= self.samples.len());
        assert!(
            self.start <= self.end,
            "start: {start:?}, end: {:?}",
            self.end
        );
    }

    pub(super) fn set_end(&mut self, end: OutFrameCount) {
        self.end = end.samples(self.channels);
        assert!(self.end.raw() <= self.samples.len());
    }

    pub(super) fn current_span_len(&self) -> usize {
        (self.end - self.start).raw()
    }
}

impl Iterator for Output {
    type Item = Sample;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.end {
            None
        } else {
            let sample = self.samples[self.pos.raw()];
            self.pos += 1usize;
            Some(sample)
        }
    }
}
