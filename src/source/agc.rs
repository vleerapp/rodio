//
//      Automatic Gain Control (AGC) Algorithm
//      Designed by @UnknownSuperficialNight
//
//   Features:
//   • Adaptive peak detection with exponential smoothing (EMA)
//   • O(1) RMS level estimation via circular buffer
//   • Combined RMS and peak limiting with adaptive slowdown
//   • Asymmetric attack/release with per-sample clamping
//   • Configurable floor value for minimum gain threshold
//   • Atomic operations support (experimental)
//   • Fast release coefficient via 3rd‑order Taylor approximation (evaluated with Horner's method)
//   • Power-of-two window sizing for efficiency
//
//   Optimised for smooth and responsive gain control
//
//   Crafted with love. Enjoy! :)
//

use super::{SeekError, SpanTracker};
use crate::math::{duration_to_coefficient, duration_to_float, fast_exp};
use crate::{ChannelCount, Float, Sample, SampleRate, Source};
use std::time::Duration;

#[cfg(feature = "tracing")]
use tracing;

#[cfg(feature = "experimental")]
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[cfg(all(feature = "experimental", not(feature = "64bit")))]
use atomic_float::AtomicF32;
#[cfg(all(feature = "experimental", feature = "64bit"))]
use atomic_float::AtomicF64;

#[cfg(all(feature = "experimental", not(feature = "64bit")))]
type AtomicFloat = AtomicF32;
#[cfg(all(feature = "experimental", feature = "64bit"))]
type AtomicFloat = AtomicF64;

/// Ensures `RMS_WINDOW_SIZE` is a power of two
const fn power_of_two(n: usize) -> usize {
    assert!(
        n.is_power_of_two(),
        "RMS_WINDOW_SIZE must be a power of two"
    );
    n
}

/// Divide `a` by `b` unless `b` is NaN, infinite, or <= 0,
/// in which case `fallback` is returned.
#[inline(always)]
fn div_or_fallback(a: Float, b: Float, fallback: Float) -> Float {
    if b.is_finite() && b > 0.0 {
        a / b
    } else {
        fallback
    }
}

/// Size of the circular buffer used for RMS calculation.
/// A larger size provides more stable RMS values but increases latency.
const RMS_WINDOW_SIZE: usize = power_of_two(1024);

/// Settings for the Automatic Gain Control (AGC).
///
/// This struct contains parameters that define how the AGC will function,
/// allowing users to customise its behaviour.
#[derive(Debug, Clone)]
pub struct AutomaticGainControlSettings {
    /// The desired output level that the AGC tries to maintain.
    /// A value of 1.0 means no change to the original level.
    pub target_level: Float,
    /// Time constant for gain increases (how quickly the AGC responds to level increases).
    /// Longer durations result in slower, more gradual gain increases.
    pub attack_time: Duration,
    /// Time constant for gain decreases (how quickly the AGC responds to level decreases).
    /// Shorter durations allow for faster response to sudden loud signals.
    pub release_time: Duration,
    /// Maximum allowable gain multiplication to prevent excessive amplification.
    /// This acts as a safety limit to avoid distortion from over-amplification.
    pub absolute_max_gain: Float,
    /// Duration of the peak tracking smoothing window.
    /// Controls how much peak level measurements are smoothed before being used for gain calculation.
    /// Larger values provide more stable peak detection but add latency to peak tracking.
    /// Smaller values respond faster to sudden peaks but may allow more transient clipping.
    pub peak_tracking_window: Duration,
}

impl Default for AutomaticGainControlSettings {
    fn default() -> Self {
        AutomaticGainControlSettings {
            target_level: 1.0,                               // Default to original level
            attack_time: Duration::from_millis(500),         // Recommended attack time
            release_time: Duration::from_micros(500),        // Recommended release time
            absolute_max_gain: 7.0,                          // Recommended max gain
            peak_tracking_window: Duration::from_millis(10), // Recommended peak tracking window for balanced stability and responsiveness
        }
    }
}

#[cfg(feature = "experimental")]
/// Automatic Gain Control filter for maintaining consistent output levels.
///
/// This struct implements an AGC algorithm that dynamically adjusts audio levels
/// based on both **peak** and **RMS** (Root Mean Square) measurements.
#[derive(Clone, Debug)]
pub struct AutomaticGainControl<I> {
    input: I,

    // Core gain values
    target_level: Arc<AtomicFloat>,
    floor: Float,
    absolute_max_gain: Arc<AtomicFloat>,
    peak_tracking_window: Duration,
    current_gain: Float,

    // Timing parameters
    attack_duration: Arc<AtomicFloat>,
    release_duration: Arc<AtomicFloat>,

    // Signal analysis state
    peak_level: Float,
    release_coefficient: Float,
    rms_window: CircularBufferRMS,

    // Control flags
    is_enabled: Arc<AtomicBool>,
    span: SpanTracker,

    // Slowdown tracking
    slow_down_state: SlowDownState,
}

#[cfg(not(feature = "experimental"))]
/// Automatic Gain Control filter for maintaining consistent output levels.
///
/// This struct implements an AGC algorithm that dynamically adjusts audio levels
/// based on both **peak** and **RMS** (Root Mean Square) measurements.
#[derive(Clone, Debug)]
pub struct AutomaticGainControl<I> {
    input: I,

    // Core gain values
    target_level: Float,
    floor: Float,
    absolute_max_gain: Float,
    peak_tracking_window: Duration,
    current_gain: Float,

    // Timing parameters
    attack_duration: Float,
    release_duration: Float,

    // Signal analysis state
    peak_level: Float,
    release_coefficient: Float,
    rms_window: CircularBufferRMS,

    // Control flags
    is_enabled: bool,
    span: SpanTracker,

    // Slowdown tracking
    slow_down_state: SlowDownState,
}

/// State for adaptive slowdown of gain changes.
///
/// This struct holds the state for managing the slowdown of gain changes based on signal conditions.
/// The `slowdown_factor` determines how quickly or slowly the gain can change:
/// - When the signal is quiet and we're close to target, changes are allowed normally
/// - When the signal peaks significantly, changes are slowed down exponentially
/// - This prevents abrupt loudness jumps during automatic gain control adjustments.
#[derive(Clone, Debug)]
struct SlowDownState {
    block_size: usize,
    sample_counter: usize,
    slowdown_factor: Float,
}

impl SlowDownState {
    #[inline]
    fn new(sample_rate: SampleRate) -> Self {
        // Calculate and cache block size based on sample rate
        let block_size = (sample_rate.get() as usize / 1000) * 2; // 2ms blocks

        Self {
            block_size,
            sample_counter: 0,
            slowdown_factor: 0.0,
        }
    }

    #[inline]
    fn increment_sample_counter(&mut self) {
        self.sample_counter = (self.sample_counter + 1) % self.block_size;
    }

    /// Computes the slowdown factor for adaptive gain changes.
    ///
    /// The slowdown factor determines how quickly or slowly the gain can change based on the current signal conditions.
    /// - When the desired gain is close to the current gain, the slowdown factor increases, preventing abrupt loudness jumps during automatic gain control adjustments.
    /// - When the signal deviates significantly from the target, the slowdown factor remains high to maintain stability.
    #[inline]
    fn compute_slowdown_factor(
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

/// A circular buffer optimised for RMS calculation over a sliding window.
///
/// Maintains a running sum of squares with O(1) updates and retrieval,
/// avoiding the need to scan stored samples for mean calculations.
#[derive(Clone, Debug)]
struct CircularBufferRMS {
    buffer: Box<[Float; RMS_WINDOW_SIZE]>,
    sum_of_squares: Float,
    index: usize,
}

impl CircularBufferRMS {
    /// Creates a new `CircularBufferRMS` with a fixed size determined at compile time.
    ///
    /// The buffer size is `RMS_WINDOW_SIZE`, chosen as a power of two for
    /// efficient modulo operations using bitwise arithmetic.
    #[inline]
    fn new() -> Self {
        CircularBufferRMS {
            buffer: Box::new([0.0; RMS_WINDOW_SIZE]),
            sum_of_squares: 0.0,
            index: 0,
        }
    }

    /// Adds a sample to the buffer and updates the running sum of squares.
    ///
    /// Maintains an incremental sum of squares for O(1) RMS computation
    /// without recalculating from stored samples.
    #[inline]
    fn push(&mut self, value: Float) {
        let old_value = self.buffer[self.index];
        // Update the sum of squares by subtracting the square of the old value and adding the square of the new value.
        self.sum_of_squares = (self.sum_of_squares - (old_value * old_value)) + (value * value);
        self.buffer[self.index] = value;
        // Use bitwise for efficient index wrapping since RMS_WINDOW_SIZE is a power of two.
        self.index = (self.index + 1) & (RMS_WINDOW_SIZE - 1);
    }

    /// Calculate the RMS (Root Mean Square) value of all values in the buffer.
    ///
    /// RMS provides a measure of the signal's effective or average magnitude.
    #[inline]
    fn rms(&self) -> Float {
        (self.sum_of_squares / RMS_WINDOW_SIZE as Float)
            .sqrt()
            .min(1.0)
    }
}

/// Constructs an `AutomaticGainControl` object with specified parameters.
///
/// # Arguments
///
/// `input` - The input audio source
/// `target_level` - The desired output level
/// `attack_time` - Time constant for gain increase
/// `release_time` - Time constant for gain decrease
/// `absolute_max_gain` - Maximum allowable gain
/// `peak_tracking_window` - Duration over which to track peak level
#[inline]
pub(crate) fn automatic_gain_control<I>(
    input: I,
    target_level: Float,
    attack_time: Duration,
    release_time: Duration,
    absolute_max_gain: Float,
    peak_tracking_window: Duration,
) -> AutomaticGainControl<I>
where
    I: Source,
{
    let sample_rate = input.sample_rate();
    let attack_duration = duration_to_float(attack_time);
    let release_duration = duration_to_float(release_time);

    let release_coefficient = duration_to_coefficient(peak_tracking_window, sample_rate);

    #[cfg(feature = "experimental")]
    {
        let channels = input.channels();
        AutomaticGainControl {
            input,
            target_level: Arc::new(AtomicFloat::new(target_level)),
            floor: 0.0,
            absolute_max_gain: Arc::new(AtomicFloat::new(absolute_max_gain)),
            peak_tracking_window,
            current_gain: 1.0,
            attack_duration: Arc::new(AtomicFloat::new(attack_duration)),
            release_duration: Arc::new(AtomicFloat::new(release_duration)),
            peak_level: 0.7,
            release_coefficient,
            rms_window: CircularBufferRMS::new(),
            is_enabled: Arc::new(AtomicBool::new(true)),
            span: SpanTracker::new(sample_rate, channels),
            slow_down_state: SlowDownState::new(sample_rate),
        }
    }

    #[cfg(not(feature = "experimental"))]
    {
        let channels = input.channels();
        AutomaticGainControl {
            input,
            target_level,
            floor: 0.0,
            absolute_max_gain,
            peak_tracking_window,
            current_gain: 1.0,
            attack_duration,
            release_duration,
            peak_level: 0.7,
            release_coefficient,
            rms_window: CircularBufferRMS::new(),
            is_enabled: true,
            span: SpanTracker::new(sample_rate, channels),
            slow_down_state: SlowDownState::new(sample_rate),
        }
    }
}

impl<I> AutomaticGainControl<I>
where
    I: Source,
{
    #[inline]
    fn target_level(&self) -> Float {
        #[cfg(feature = "experimental")]
        {
            self.target_level.load(Ordering::Relaxed)
        }
        #[cfg(not(feature = "experimental"))]
        {
            self.target_level
        }
    }

    #[inline]
    fn absolute_max_gain(&self) -> Float {
        #[cfg(feature = "experimental")]
        {
            self.absolute_max_gain.load(Ordering::Relaxed)
        }
        #[cfg(not(feature = "experimental"))]
        {
            self.absolute_max_gain
        }
    }

    #[inline]
    fn attack_duration(&self) -> Float {
        #[cfg(feature = "experimental")]
        {
            self.attack_duration.load(Ordering::Relaxed)
        }
        #[cfg(not(feature = "experimental"))]
        {
            self.attack_duration
        }
    }

    #[inline]
    fn release_duration(&self) -> Float {
        #[cfg(feature = "experimental")]
        {
            self.release_duration.load(Ordering::Relaxed)
        }
        #[cfg(not(feature = "experimental"))]
        {
            self.release_duration
        }
    }

    #[inline]
    fn is_enabled(&self) -> bool {
        #[cfg(feature = "experimental")]
        {
            self.is_enabled.load(Ordering::Relaxed)
        }
        #[cfg(not(feature = "experimental"))]
        {
            self.is_enabled
        }
    }

    #[cfg(feature = "experimental")]
    /// Access the target output level for real-time adjustment.
    ///
    /// Use this to dynamically modify the AGC's target level while audio is processing.
    /// Adjust this value to control the overall output amplitude of the processed signal.
    #[inline]
    pub fn get_target_level(&self) -> Arc<AtomicFloat> {
        Arc::clone(&self.target_level)
    }

    #[cfg(feature = "experimental")]
    /// Access the maximum gain limit for real-time adjustment.
    ///
    /// Use this to dynamically modify the AGC's maximum allowable gain during runtime.
    /// Adjusting this value helps prevent excessive amplification in low-level signals.
    #[inline]
    pub fn get_absolute_max_gain(&self) -> Arc<AtomicFloat> {
        Arc::clone(&self.absolute_max_gain)
    }

    #[cfg(feature = "experimental")]
    /// Access the attack coefficient for real-time adjustment.
    ///
    /// Use this to dynamically modify how quickly the AGC responds to level increases.
    /// Smaller values result in faster response, larger values in slower response.
    /// Adjust during runtime to fine-tune AGC behavior for different audio content.
    ///
    /// Note: if the sample rate or channel count changes, any value set through this handle will
    /// be overwritten with the attack time that this AGC was constructed with.
    #[inline]
    pub fn get_attack_duration(&self) -> Arc<AtomicFloat> {
        Arc::clone(&self.attack_duration)
    }

    #[cfg(feature = "experimental")]
    /// Access the release coefficient for real-time adjustment.
    ///
    /// Use this to dynamically modify how quickly the AGC responds to level decreases.
    /// Smaller values result in faster response, larger values in slower response.
    /// Adjust during runtime to optimize AGC behavior for varying audio dynamics.
    ///
    /// Note: if the sample rate or channel count changes, any value set through this handle will
    /// be overwritten with the release time that this AGC was constructed with.
    #[inline]
    pub fn get_release_duration(&self) -> Arc<AtomicFloat> {
        Arc::clone(&self.release_duration)
    }

    #[cfg(feature = "experimental")]
    /// Access the AGC on/off control.
    /// Use this to dynamically enable or disable AGC processing during runtime.
    ///
    /// AGC is on by default. `false` is disabled state, `true` is enabled.
    /// In disabled state the sound is passed through AGC unchanged.
    ///
    /// In particular, this control is useful for comparing processed and unprocessed audio.
    #[inline]
    pub fn get_agc_control(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.is_enabled)
    }

    /// Enable or disable AGC processing.
    ///
    /// Use this to enable or disable AGC processing.
    /// Useful for comparing processed and unprocessed audio or for disabling/enabling AGC.
    #[inline]
    pub fn set_enabled(&mut self, enabled: bool) {
        #[cfg(feature = "experimental")]
        {
            self.is_enabled.store(enabled, Ordering::Relaxed);
        }
        #[cfg(not(feature = "experimental"))]
        {
            self.is_enabled = enabled;
        }
    }

    /// Set the floor value for the AGC
    ///
    /// This method sets the floor value for the AGC. The floor value is the minimum
    /// gain that the AGC will allow. The gain will not drop below this value.
    ///
    /// Passing `None` will disable the floor value (setting it to 0.0), allowing the
    /// AGC gain to drop to very low levels.
    #[inline]
    pub fn set_floor(&mut self, floor: Option<Float>) {
        self.floor = floor.unwrap_or(0.0);
    }

    /// Updates the peak level using exponential smoothing (EMA) to blend the current
    /// value toward the previous level using the release coefficient, then taking
    /// the maximum of the current sample.
    ///
    /// This provides a stable peak measurement that doesn't react to every sample,
    /// preventing excessive gain adjustments when the signal is momentarily loud.
    /// The peak serves as an absolute maximum safeguard to prevent output clipping
    /// even when RMS-based gain calculations suggest aggressive amplification.
    #[inline]
    fn update_peak_level(&mut self, sample_value: Float, release_coefficient: Float) {
        // Compute the exponentially smoothed estimate of the previous peak level.
        // The EMA smooths peak tracking over time, preventing sudden jumps when
        // loud transients occur, which would otherwise cause extreme gain reductions.
        let peak_release =
            self.peak_level * release_coefficient + sample_value * (1.0 - release_coefficient);

        // Take maximum to ensure the peak is always an upper bound.
        // This guarantees that peak_level never decreases below the current sample,
        // preserving the safety mechanism against clipping.
        // Set an upper bound to prevent tracking out-of-bounds samples from decoders like libopus
        self.peak_level = sample_value.max(peak_release).min(1.0);
    }

    /// Updates the RMS (Root Mean Square) level using a circular buffer approach.
    /// This method calculates a moving average of the squared input samples,
    /// providing a measure of the signal's average power over time.
    #[inline]
    fn update_rms(&mut self, sample_value: Sample) -> Float {
        self.rms_window.push(sample_value);

        // Calculate RMS safely
        let rms = self.rms_window.rms();
        if rms.is_nan() || rms <= 0.0 {
            0.0 // Default to 0 if RMS is invalid
        } else {
            rms
        }
    }

    #[inline]
    fn process_sample(&mut self, sample: I::Item) -> I::Item {
        // Cache atomic loads at the start - avoids repeated atomic operations
        let target_level = self.target_level();
        let absolute_max_gain = self.absolute_max_gain();
        let attack_time_in_seconds = self.attack_duration();
        let release_duration = self.release_duration();
        let sample_rate = self.sample_rate().get() as Float; // Sample rate in Hz

        // Convert the sample to its absolute float value for level calculations
        // We use abs() to work with signal magnitude regardless of polarity
        // This is crucial because RMS and peak detection care about energy,
        // not whether the signal is positive or negative
        let sample_value = sample.abs();

        // Increment the sample counter
        self.slow_down_state.increment_sample_counter();

        // Dynamically adjust peak level using cached release coefficient
        self.update_peak_level(sample_value, self.release_coefficient);

        // Calculate the current RMS (Root Mean Square) level using a sliding window approach
        let rms = self.update_rms(sample_value);

        // Compute the gain adjustment required to reach the adjusted target level
        // When rms is 0.0 (silence), we fall back to current_gain as the default
        // This keeps the gain stable during silence without any hard reset
        // The gain will only change gradually when peaks occur or signal returns
        let rms_gain = div_or_fallback(target_level, rms, self.current_gain);

        // Calculate gain adjustments based on peak levels
        // We divide target_level by peak_level to find the gain multiplier needed
        // to scale the signal's peaks to match the target. If peak_level is high
        // (loud signal), this gives us a gain < 1.0 (attenuation). If peak_level
        // is low (quiet signal), this gives us a gain > 1.0 (amplification).
        // The peak level acts as a safety mechanism to prevent output spikes
        // that could exceed the target level.
        let peak_gain = div_or_fallback(target_level, self.peak_level, 1.0).min(absolute_max_gain);

        // Combine RMS and peak gains by taking the minimum. We use min() because
        // we need to choose a single gain value that respects both constraints.
        // Think of it like this: RMS gain might suggest "amplify by 5x" based on
        // average signal level, but peak gain might suggest "attenuate by 0.5x"
        // to prevent output spikes. Since these goals conflict (amplify vs reduce),
        // we pick the more conservative one: min() selects 0.5x (attenuation) over
        // 5x (amplification). This ensures we don't blindly amplify and risk
        // output spikes, even when the average signal seems quiet.
        // Then we apply the floor to ensure we never drop below the minimum allowed gain.
        let desired_gain = rms_gain.min(peak_gain).max(self.floor);

        if self.slow_down_state.sample_counter == 0 {
            self.slow_down_state.compute_slowdown_factor(
                desired_gain,
                self.current_gain,
                rms,
                self.peak_level,
            );
        }

        let dynamic_attack_time = attack_time_in_seconds * self.slow_down_state.slowdown_factor;

        // Calculate max gain change per sample based on dynamic attack/release times
        let max_attack_gain_change_per_sample = 1.0 / (dynamic_attack_time * sample_rate);
        let max_release_gain_change_per_sample = 1.0 / (release_duration * sample_rate);

        // Determine gain difference
        let gain_diff = desired_gain - self.current_gain;

        // Clamp gain change based on attack or release phase
        let gain_change = if gain_diff > 0.0 {
            // Attack phase: Clamp the gain change to the maximum allowed per sample
            gain_diff.clamp(0.0, max_attack_gain_change_per_sample)
        } else {
            // Release phase: Clamp the gain change to the maximum allowed per sample
            gain_diff.clamp(-max_release_gain_change_per_sample, 0.0)
        };

        // Update current gain
        self.current_gain += gain_change;

        #[cfg(feature = "tracing")]
        if self.slow_down_state.sample_counter == 0 {
            tracing::debug!(
            "RMS: {:.4}, Peak: {:.4}, Desired Gain: {:.4}, Current Gain: {:.4}, Release Coefficient: {}, Attack Time: {:.4}",
            rms, self.peak_level, desired_gain, self.current_gain, self.release_coefficient, dynamic_attack_time,
        );
        }

        // Apply gain to sample and return
        sample * self.current_gain
    }

    /// Returns an immutable reference to the inner source.
    pub fn inner(&self) -> &I {
        &self.input
    }

    /// Returns a mutable reference to the inner source.
    pub fn inner_mut(&mut self) -> &mut I {
        &mut self.input
    }
}

impl<I> Iterator for AutomaticGainControl<I>
where
    I: Source,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let detection = self.span.advance(&self.input);

        if detection.at_span_boundary && detection.parameters_changed {
            let current_sample_rate = self.input.sample_rate();

            // Recalculate coefficients for new sample rate
            self.release_coefficient =
                duration_to_coefficient(self.peak_tracking_window, current_sample_rate);

            // Reset RMS window to avoid mixing samples from different parameter sets
            self.rms_window = CircularBufferRMS::new();
            self.peak_level = 0.7;
            self.current_gain = 1.0;
        }

        let sample = self.input.next()?;

        let output = if self.is_enabled() {
            self.process_sample(sample)
        } else {
            sample
        };
        Some(output)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.input.size_hint()
    }
}

impl<I> ExactSizeIterator for AutomaticGainControl<I> where I: Source + ExactSizeIterator {}

impl<I> Source for AutomaticGainControl<I>
where
    I: Source,
{
    #[inline]
    fn current_span_len(&self) -> Option<usize> {
        self.input.current_span_len()
    }

    #[inline]
    fn channels(&self) -> ChannelCount {
        self.input.channels()
    }

    #[inline]
    fn sample_rate(&self) -> SampleRate {
        self.input.sample_rate()
    }

    #[inline]
    fn total_duration(&self) -> Option<Duration> {
        self.input.total_duration()
    }

    #[inline]
    fn try_seek(&mut self, pos: Duration) -> Result<(), SeekError> {
        self.input.try_seek(pos)?;
        self.span.seek(pos, &self.input);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::nz;
    use crate::source::test_utils::TestSource;
}
