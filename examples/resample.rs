//! Example demonstrating audio resampling with different quality presets.

use clap::Parser;
use rodio::source::{ResampleConfig, Source};
#[cfg(feature = "wav_output")]
use rodio::wav_to_file;
use rodio::{Decoder, DeviceSinkBuilder, Player};
use std::error::Error;
use std::num::NonZero;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(about = "Resample audio using different quality presets")]
struct Args {
    /// Target sample rate in Hz (default: device native rate when playing, source rate when
    /// writing)
    #[arg(long = "rate")]
    target_rate: Option<NonZero<u32>>,

    /// Path to input audio file
    #[arg(long = "input", default_value = "assets/music.ogg")]
    input: PathBuf,

    /// Path to output WAV file; if omitted, audio plays to the default device
    #[cfg(feature = "wav_output")]
    #[arg(long = "output")]
    output: Option<PathBuf>,

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

    let file = std::fs::File::open(&args.input)
        .map_err(|e| format!("Failed to open '{}': {e}", args.input.display()))?;
    let source = Decoder::try_from(file)?;

    let source_rate = source.sample_rate().get();
    let channels = source.channels().get();

    if let Some(dur) = source.total_duration() {
        println!("Duration: {dur:?}");
    }

    #[cfg(feature = "wav_output")]
    if let Some(output_path) = args.output {
        let target_rate = args.target_rate.unwrap_or_else(|| source.sample_rate());
        println!("Resampling {channels}ch {source_rate} Hz → {target_rate} Hz");
        println!("Configuration: {config:#?}");
        let resampled = source.resample(target_rate, config);
        println!("Writing to '{}'...", output_path.display());
        let start = Instant::now();
        wav_to_file(resampled, &output_path)?;
        println!("Finished in {:?}", start.elapsed());
        return Ok(());
    }

    let builder = DeviceSinkBuilder::from_default_device()?;
    let stream_handle = match args.target_rate {
        Some(rate) => builder.with_sample_rate(rate).open_stream()?,
        None => builder.open_stream()?,
    };
    let target_rate = stream_handle.config().sample_rate();

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
