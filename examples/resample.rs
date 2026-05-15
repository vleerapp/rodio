//! Example demonstrating audio resampling with different quality presets.

use clap::Parser;
use rodio::source::{ResampleConfig, Source};
use rodio::{Decoder, DeviceSinkBuilder, Player};
use std::error::Error;
use std::num::NonZero;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(about = "Resample audio using different quality presets")]
struct Args {
    /// Target sample rate in Hz (default: device native rate)
    #[arg(long = "rate")]
    target_rate: Option<NonZero<u32>>,

    /// Path to audio file
    #[arg(long = "file", default_value = "assets/music.ogg")]
    audio_file: PathBuf,

    /// Resampling method
    #[arg(long = "method", value_enum, default_value_t = Method::Balanced)]
    method: Method,
}

#[derive(clap::ValueEnum, Clone)]
enum Method {
    /// Nearest-neighbor (zero-order hold) polynomial resampling. Fastest, no anti-aliasing.
    Nearest,
    /// Linear polynomial resampling. Fast, no anti-aliasing.
    Linear,
    /// Cubic polynomial resampling. Smoother than linear, no anti-aliasing.
    Cubic,
    /// Quintic polynomial resampling. Smoother than cubic, no anti-aliasing.
    Quintic,
    /// Septic polynomial resampling. Highest polynomial quality, no anti-aliasing.
    Septic,
    /// 64-tap sinc, linear interpolation, Hann2 window.
    VeryFast,
    /// 128-tap sinc, linear interpolation, Blackman2 window.
    Fast,
    /// 192-tap sinc, quadratic interpolation, BlackmanHarris2 window (default).
    Balanced,
    /// 256-tap sinc, cubic interpolation, BlackmanHarris2 window.
    Accurate,
}

impl From<Method> for ResampleConfig {
    fn from(method: Method) -> Self {
        match method {
            Method::Nearest => ResampleConfig::nearest(),
            Method::Linear => ResampleConfig::linear(),
            Method::Cubic => ResampleConfig::cubic(),
            Method::Quintic => ResampleConfig::quintic(),
            Method::Septic => ResampleConfig::septic(),
            Method::VeryFast => ResampleConfig::very_fast(),
            Method::Fast => ResampleConfig::fast(),
            Method::Balanced => ResampleConfig::balanced(),
            Method::Accurate => ResampleConfig::accurate(),
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let config = ResampleConfig::from(args.method);

    let builder = DeviceSinkBuilder::from_default_device()?;
    let stream_handle = match args.target_rate {
        Some(rate) => builder.with_sample_rate(rate).open_stream()?,
        None => builder.open_stream()?,
    };
    let target_rate = stream_handle.config().sample_rate();

    let file = std::fs::File::open(&args.audio_file)
        .map_err(|e| format!("Failed to open '{}': {e}", args.audio_file.display()))?;
    let source = Decoder::try_from(file)?;

    let source_rate = source.sample_rate().get();
    let channels = source.channels().get();

    if let Some(dur) = source.total_duration() {
        println!("Duration: {dur:?}");
    }

    println!("Resampling {channels}ch {source_rate} Hz → {target_rate} Hz");
    println!("Configuration: {config:#?}");

    let resampled = source.resample(target_rate, config);
    let player = Player::connect_new(stream_handle.mixer());

    println!("Playing... (Ctrl+C to stop)");
    let start = Instant::now();
    player.append(resampled);
    player.sleep_until_end();

    println!("Finished in {:?}", start.elapsed());
    Ok(())
}
