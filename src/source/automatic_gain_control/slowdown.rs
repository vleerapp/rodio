use super::div_or_fallback;
use crate::math::fast_exp;
use crate::{Float, SampleRate};

/// State for adaptive slowdown of gain changes.
///
/// This struct holds the state for managing the slowdown of gain changes based on signal conditions.
/// The `slowdown_factor` determines how quickly or slowly the gain can change:
/// - When the signal is quiet and we're close to target, changes are allowed normally
/// - When the signal peaks significantly, changes are slowed down exponentially
/// - This prevents abrupt loudness jumps during automatic gain control adjustments.
#[derive(Clone, Debug)]
pub(super) struct SlowDownState {
    block_size: usize,
    pub(super) sample_counter: usize,
    pub(super) slowdown_factor: Float,
}

impl SlowDownState {
    #[inline]
    pub(super) fn new(sample_rate: SampleRate) -> Self {
        // Calculate and cache block size based on sample rate
        let block_size = (sample_rate.get() as usize / 1000) * 2; // 2ms blocks

        Self {
            block_size,
            sample_counter: 0,
            slowdown_factor: 0.0,
        }
    }

    #[inline]
    pub(super) fn increment_sample_counter(&mut self) {
        self.sample_counter = (self.sample_counter + 1) % self.block_size;
    }

    /// Computes the slowdown factor for adaptive gain changes.
    ///
    /// The slowdown factor determines how quickly or slowly the gain can change based on the current signal conditions.
    /// - When the desired gain is close to the current gain, the slowdown factor increases, preventing abrupt loudness jumps during automatic gain control adjustments.
    /// - When the signal deviates significantly from the target, the slowdown factor remains high to maintain stability.
    #[inline]
    pub(super) fn compute_slowdown_factor(
        &mut self,
        desired_gain: Float,
        current_gain: Float,
        rms: Float,
        peak_level: Float,
    ) {
        // Calculate the absolute difference between the desired gain and the current gain
        let distance_from_target = (desired_gain - current_gain).abs();

        // Calculate the maximum distance as the sum of RMS and peak level
        let max_distance = rms + peak_level;

        // Normalise distance clamped between [0,1] with a fallback of 1.0
        let normalise_distance = div_or_fallback(distance_from_target, max_distance, 1.0).min(1.0);

        // Compute the exponential slowdown factor based on the normalised distance
        // The multiplier is scaled by the square root of the sum of peak level and RMS
        let exp_multiplier = 10.0 * (peak_level + rms).sqrt();
        let exp_slowdown = fast_exp(1.0 + exp_multiplier * (1.0 - normalise_distance));

        // Create a mask that is 1.0 if the distance is within the max_distance, otherwise 0.0
        // This mask is used to blend the exponential slowdown factor with a linear factor
        let mask = ((max_distance - distance_from_target).max(0.0) / max_distance).min(1.0);

        // Blend the slowdown factor: when mask=1 use exp_slowdown, else 1.0
        // This ensures that the slowdown factor increases when the signal deviates from the target
        self.slowdown_factor = 1.0 + mask * (exp_slowdown - 1.0);
    }
}
