use super::{SeekError, SpanTracker};

mod agc;
mod config;
mod helpers;
mod rms;
mod slowdown;

pub use agc::AutomaticGainControl;
pub use config::AutomaticGainControlSettings;

use helpers::div_or_fallback;
use rms::CircularBufferRMS;
use slowdown::SlowDownState;
