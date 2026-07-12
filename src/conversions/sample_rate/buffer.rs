//! Fixed-capacity sample buffer with a read cursor.
//!
use std::fmt::{Debug, Write};

use crate::common::{FrameCount, OutFrameCount, OutSamples, SampleCount};
use crate::{ChannelCount, Sample, SampleRate};

pub(crate) struct Output {
    start: OutSamples,
    pos: OutSamples,
    end: OutSamples,

    data: Box<[Sample]>,
    pub channels: ChannelCount,
    pub source_rate: SampleRate,
}

impl Debug for Output {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Output")
            .field("start", &self.start)
            .field("pos", &self.pos)
            .field("end", &self.end)
            .field("data", &LimitLength(&self.data))
            .field("channels", &self.channels)
            .field("source_rate", &self.source_rate)
            .finish()
    }
}

struct LimitLength<'a>(&'a [Sample]);

impl Debug for LimitLength<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[")?;
        for element in self.0.iter().take(3) {
            f.write_str("\n\t")?;
            f.write_fmt(format_args!("{element:?}"))?;
            f.write_char(',')?;
        }

        if let Some(hidden) = self.0.len().checked_sub(6) {
            f.write_str("\n\t... ")?;
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
        let mut data = Vec::new();
        data.reserve_exact(capacity.samples(channels).raw());
        data.resize(data.capacity(), 0.0);
        Self {
            start: OutSamples::ZERO,
            pos: OutSamples::ZERO,
            end: OutSamples::ZERO,
            data: data.into_boxed_slice(),
            channels,
            source_rate,
        }
    }

    pub(super) fn capacity(&self) -> FrameCount {
        SampleCount(self.data.len()).frames(self.channels)
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
        &mut self.data
    }

    pub(super) fn set_start(&mut self, start: OutFrameCount) {
        self.start = start.samples(self.channels);
        self.pos = self.start;
        assert!(self.start.raw() <= self.data.len());
        assert!(
            self.start <= self.end,
            "start: {start:?}, end: {:?}",
            self.end
        );
    }

    pub(super) fn set_end(&mut self, end: OutFrameCount) {
        self.end = end.samples(self.channels);
        assert!(self.end.raw() <= self.data.len());
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
            let sample = self.data[self.pos.raw()];
            self.pos += 1usize;
            Some(sample)
        }
    }
}
