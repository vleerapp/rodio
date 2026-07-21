use std::{num::NonZeroU32, time::Duration};

use crate::Float;

/// A circular buffer optimised for RMS calculation over a sliding window.
///
/// Maintains a running sum of squares with O(1) updates and retrieval,
/// avoiding the need to scan stored samples for mean calculations.
#[derive(Clone, Debug)]
pub(super) struct CircularBufferRMS {
    buffer: Box<[Float]>, // Runtime-sized window so RMS spans the same time range at any sample rate
    sum_of_squares: Float, // Keeps a running square-sum so RMS can be updated without re-scanning the entire buffer
    index: usize, // Marks the current slot; each new sample overwrites the oldest one as we advance
    mask: usize, // Lets the index wrap with `&` instead of `%`, which is faster because the size is a power of two
    reciprocal_len: Float, // Stores `1 / len` so RMS normalizes with multiplication instead of division
}

impl CircularBufferRMS {
    /// Calculates the buffer size from the sample rate and target window length.
    ///
    /// The window is expressed in milliseconds, converted to samples, then rounded
    /// up to the next power of two for efficient index wrapping using bitwise arithmetic.
    #[inline]
    fn calculate_rms_buffer_size(sample_rate: NonZeroU32, window_ms: Duration) -> usize {
        // Convert the time window into the number of samples for this sample rate
        // Example: 44,100 × 20 ms / 1,000 -> 882 samples
        let samples = (sample_rate.get() as usize * window_ms.as_millis() as usize).div_ceil(1000);

        // Ensure minimum 1 sample, then round up to nearest power of two
        // Result: 882 samples -> 1024-sample buffer
        // Which is: 1024 samples at 44,100 Hz ≈ 23.2 ms
        samples.max(1).next_power_of_two()
    }

    /// Creates a new `CircularBufferRMS` from a sample rate and target window size in milliseconds.
    ///
    /// The buffer size is computed from the requested duration and rounded up to a power of two
    /// so wrapping can use bitwise arithmetic instead of modulo.
    #[inline]
    pub(super) fn new(sample_rate: NonZeroU32, window_ms: Duration) -> Self {
        // Calculate the buffer size from the sample_rate and target window
        let size = Self::calculate_rms_buffer_size(sample_rate, window_ms);

        CircularBufferRMS {
            buffer: vec![0.0; size].into_boxed_slice(), // [T; N] requires const N; Vec allows runtime size
            sum_of_squares: 0.0,
            index: 0,
            mask: size - 1,
            reciprocal_len: 1.0 / size as Float,
        }
    }

    /// Adds a sample to the buffer and updates the running sum of squares.
    ///
    /// Maintains an incremental sum of squares for O(1) RMS computation
    /// without recalculating from stored samples.
    #[inline]
    pub(super) fn push(&mut self, value: Float) {
        let old_value = self.buffer[self.index];
        // Update the sum of squares by subtracting the square of the old value and adding the square of the new value.
        self.sum_of_squares = (self.sum_of_squares - (old_value * old_value)) + (value * value);
        self.buffer[self.index] = value;
        // Use bitwise for efficient index wrapping since the buffer size is a power of two.
        self.index = (self.index + 1) & self.mask;
    }

    /// Calculate the RMS (Root Mean Square) value of all values in the buffer.
    ///
    /// RMS provides a measure of the signal's effective or average magnitude.
    #[inline]
    pub(super) fn rms(&self) -> Float {
        (self.sum_of_squares * self.reciprocal_len).sqrt()
    }
}
