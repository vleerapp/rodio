use crate::Float;
use std::time::Duration;

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
    /// The minimum output level (gain floor) that the AGC will not go below.
    /// A value of 1.0 preserves loud passages at source level without additional amplification (amplification only).
    /// A value of 0.0 allows unlimited amplification (pure AGC behaviour).
    pub floor: Float,
}

impl AutomaticGainControlSettings {
    /// Returns a preset optimised for music content.
    ///
    /// Values tuned through empirical testing and are intended as good defaults for general music processing.
    pub fn music_preset() -> Self {
        AutomaticGainControlSettings {
            target_level: 1.0,
            attack_time: Duration::from_millis(500),
            release_time: Duration::from_micros(500),
            absolute_max_gain: 7.0,
            peak_tracking_window: Duration::from_millis(10),
            floor: 1.0,
        }
    }

    /// Returns a preset optimised for speech content.
    ///
    /// Values tuned through empirical testing and are intended as good defaults for general speech processing.
    pub fn speech_preset() -> Self {
        AutomaticGainControlSettings {
            target_level: 1.0,
            attack_time: Duration::from_millis(250),
            release_time: Duration::from_micros(50),
            absolute_max_gain: 7.0,
            peak_tracking_window: Duration::from_millis(10),
            floor: 0.0,
        }
    }
}

impl Default for AutomaticGainControlSettings {
    // Music preset is the default
    fn default() -> Self {
        Self::music_preset()
    }
}
