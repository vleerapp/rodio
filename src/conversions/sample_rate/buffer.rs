//! Fixed-capacity sample buffer with a read cursor.
//!
//! Holds one chunk of resampled output. Callers reset it with the number of freshly written
//! samples, optionally skip delay samples at the head, optionally cap it to trim filter
//! artifacts at the tail, and then drain it sample-by-sample.

use std::fmt;

use crate::common::OutSamples;
use crate::Sample;
use dasp_sample::Sample as _;

/// Fixed-capacity sample buffer with a read cursor.
pub(crate) struct OutputBuffer {
    data: Box<[Sample]>,
    pos: OutSamples,
    len: OutSamples,
}

impl fmt::Debug for OutputBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OutputBuffer")
            .field("capacity", &self.data.len())
            .field("pos", &self.pos)
            .field("len", &self.len)
            .finish()
    }
}

impl OutputBuffer {
    /// Create a new buffer with the given capacity, initialized to equilibrium samples.
    pub(crate) fn new(capacity: OutSamples) -> Self {
        Self {
            data: vec![Sample::EQUILIBRIUM; capacity.raw()].into_boxed_slice(),
            pos: OutSamples::ZERO,
            len: OutSamples::ZERO,
        }
    }

    /// Reset for a new fill: rewind cursor to 0 and record the number of valid samples.
    pub(crate) fn rewind_to(&mut self, filled: OutSamples) {
        self.pos = OutSamples::ZERO;
        self.len = filled;
    }

    /// Advance the cursor by `n` samples (capped at `len`).
    /// returns items skipped
    pub(crate) fn skip(&mut self, n: OutSamples) -> OutSamples {
        let n = n.min(self.remaining());
        self.pos += n;
        n
    }

    /// Shrink `len` so at most `remaining` more samples will be returned from the cursor.
    pub(crate) fn cap_to_remaining(&mut self, remaining: OutSamples) {
        self.len = self.len.min(self.pos + remaining);
    }

    /// True when the cursor has reached the end of the valid data.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.pos >= self.len
    }

    /// Read the next sample and advance the cursor. Panics in debug if the buffer is empty.
    #[inline]
    pub(crate) fn read(&mut self) -> Sample {
        debug_assert!(!self.is_empty(), "read from empty Buffer");
        let s = self.data[self.pos.raw()];
        self.pos += 1;
        s
    }

    /// Total capacity of the backing allocation.
    pub(crate) fn capacity(&self) -> OutSamples {
        OutSamples(self.data.len())
    }

    /// Number of samples remaining before the cursor reaches the end.
    #[inline]
    pub(crate) fn remaining(&self) -> OutSamples {
        self.len - self.pos
    }

    /// Full backing slice for writing via an audio adapter.
    pub(crate) fn as_mut_slice(&mut self) -> &mut [Sample] {
        &mut self.data
    }
}
