//! Example to acquire microphone data, and measure and plot the interval between samples.
//! This should be run in release mode.

use kuva::prelude::*;
use rodio::microphone::MicrophoneBuilder;
use rodio::Source;
use std::error::Error;
use std::path::Path;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let mut input = MicrophoneBuilder::new()
        .default_device()?
        .default_config()?
        .prefer_sample_rates(vec![44_100.try_into()?, 48_000.try_into()?])
        .prefer_buffer_sizes(vec![16, 32, 64, 128, 256])
        .open_stream()?;
    let config = *input.config();
    println!("Device config: {config:#?}");

    let duration = Duration::from_secs(5);
    println!("Recording for {duration:#?}...");
    let n_samples_est = input.sample_rate().get() as usize
        * duration.as_secs() as usize
        * input.channels().get() as usize;
    let mut intervals = Vec::with_capacity(n_samples_est);
    let start = Instant::now();
    let mut last_t = start;
    while start.elapsed() < duration {
        let _sample = input.next().unwrap();
        let dt = last_t.elapsed();
        last_t = Instant::now();
        intervals.push(dt);
    }
    println!(
        "Recorded {} samples; mean sampling rate: {:.2} Hz ({:.2} μs/sample)",
        intervals.len(),
        intervals.len() as f64 / duration.as_secs_f64(),
        duration.as_secs_f64() / intervals.len() as f64 * 1e6,
    );
    print_stats(&intervals);

    println!("Rendering plots...");
    plot_intervals(
        &intervals,
        "microphone_intervals.png",
        format!("{config:?}"),
    )?;
    println!("Plots saved to microphone_intervals.png");

    /*
    let csv_file_path = "microphone_intervals.csv";
    let mut csv_file = std::fs::File::create(csv_file_path)?;
    use std::io::Write;
    for dt in &intervals {
        writeln!(csv_file, "{}", dt.as_nanos())?;
    }
    println!("Saved intervals to {}", csv_file_path);
    */
    Ok(())
}

fn print_stats(intervals: &[Duration]) {
    let mean = intervals.iter().sum::<Duration>() / intervals.len() as u32;
    let std = intervals
        .iter()
        .map(|dt| (dt.saturating_sub(mean).as_nanos() as f64).powf(2.0))
        .sum::<f64>()
        / intervals.len() as f64;
    let std = Duration::from_nanos(std.sqrt() as u64);
    let max = intervals.iter().max().unwrap();
    let min = intervals.iter().min().unwrap();
    println!("Measured interval between samples:");
    println!(" mean: {:.2} μs", mean.as_nanos() as f64 / 1000.0);
    println!(" std dev: {:.2} μs", std.as_nanos() as f64 / 1000.0);
    println!(" max: {:.2} μs", max.as_nanos() as f64 / 1000.0);
    println!(" min: {:.2} μs", min.as_nanos() as f64 / 1000.0);
}

fn plot_intervals(
    intervals: &[Duration],
    dest: impl AsRef<Path>,
    title: String,
) -> Result<(), Box<dyn Error>> {
    let durations = intervals.iter().map(|dt| dt.as_nanos() as f64 / 1000.0);
    let data_x_y = durations.clone().enumerate().map(|(i, dt)| (i as f64, dt));
    let scatter = ScatterPlot::new()
        .with_data(data_x_y)
        .with_marker(MarkerShape::Plus)
        .with_color("steelblue")
        .into();
    let scatter_plot = vec![scatter];
    let scatter_layout = Layout::auto_from_plots(&scatter_plot)
        .with_x_label("Sample number")
        .with_y_label("Interval [μs]");

    let histogram = Histogram::new()
        .with_data(durations)
        .with_bins(100)
        .with_color("steelblue")
        .into();
    let hist_plot = vec![histogram];
    let hist_layout = Layout::auto_from_plots(&hist_plot)
        .with_log_y()
        .with_x_label("Interval [μs]")
        .with_y_label("Count");

    let scene = Figure::new(1, 2)
        .with_plots(vec![scatter_plot, hist_plot])
        .with_layouts(vec![scatter_layout, hist_layout])
        .with_labels()
        .with_title(title)
        .with_title_size(10)
        .render();

    let png = PngBackend::new().with_scale(4.0).render_scene(&scene)?;
    std::fs::write(dest, &png[..])?;
    Ok(())
}
