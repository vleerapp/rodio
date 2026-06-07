//! Fixed-capacity sample buffer with a read cursor.
//!
//! Holds one chunk of resampled output. Callers reset it with the number of freshly written
//! samples, optionally skip delay samples at the head, optionally cap it to trim filter
//! artifacts at the tail, and then drain it sample-by-sample.

use std::fmt;

use crate::Sample;
use dasp_sample::Sample as _;

/// Fixed-capacity sample buffer with a read cursor.
pub struct Buffer {
    data: Box<[Sample]>,
    pos: usize,
    len: usize,
}

impl fmt::Debug for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Buffer")
            .field("capacity", &self.data.len())
            .field("pos", &self.pos)
            .field("len", &self.len)
            .finish()
    }
}

impl Buffer {
    /// Create a new buffer with the given capacity, initialized to equilibrium samples.
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![Sample::EQUILIBRIUM; capacity].into_boxed_slice(),
            pos: 0,
            len: 0,
        }
    }

    /// Reset for a new fill: rewind cursor to 0 and record the number of valid samples.
    pub fn reset(&mut self, filled: usize) {
        self.pos = 0;
        self.len = filled;
    }

    /// Advance the cursor by `n` samples (capped at `len`).
    pub fn skip(&mut self, n: usize) {
        self.pos = (self.pos + n).min(self.len);
    }

    /// Shrink `len` so at most `remaining` more samples will be returned from the cursor.
    pub fn cap_to_remaining(&mut self, remaining: usize) {
        self.len = self.len.min(self.pos + remaining);
    }

    /// True when the cursor has reached the end of the valid data.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pos >= self.len
    }

    /// Read the next sample and advance the cursor. Panics in debug if the buffer is empty.
    #[inline]
    pub fn read(&mut self) -> Sample {
        debug_assert!(!self.is_empty(), "read from empty Buffer");
        let s = self.data[self.pos];
        self.pos += 1;
        s
    }

    /// Total capacity of the backing allocation.
    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Number of valid samples set by the last `reset` call.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Number of samples remaining before the cursor reaches the end.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.len - self.pos
    }

    /// Full backing slice for writing via an audio adapter.
    pub fn as_mut_slice(&mut self) -> &mut [Sample] {
        &mut self.data
    }
}
