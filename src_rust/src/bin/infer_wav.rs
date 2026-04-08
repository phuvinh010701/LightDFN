use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use lightdfn_wasm::LightDFNProcessor;

const HOP_SIZE: usize = 480;

fn read_wav_mono_f32(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open wav: {}", path.display()))?;
    let spec = reader.spec();

    if spec.sample_rate != 48_000 {
        bail!(
            "expected 48kHz input, got {} Hz (file: {})",
            spec.sample_rate,
            path.display()
        );
    }

    let channels = spec.channels as usize;
    if channels == 0 {
        bail!("invalid channel count 0");
    }

    let interleaved: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("failed reading i16 samples")?,
        (hound::SampleFormat::Int, 24 | 32) => reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / i32::MAX as f32))
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("failed reading i32 samples")?,
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("failed reading f32 samples")?,
        _ => bail!(
            "unsupported wav format: {:?} {}-bit",
            spec.sample_format,
            spec.bits_per_sample
        ),
    };

    if channels == 1 {
        return Ok((interleaved, spec.sample_rate));
    }

    let mut mono = Vec::with_capacity(interleaved.len() / channels);
    for frame in interleaved.chunks_exact(channels) {
        let sum: f32 = frame.iter().copied().sum();
        mono.push(sum / channels as f32);
    }

    Ok((mono, spec.sample_rate))
}

fn write_wav_mono_i16(path: &Path, sr: u32, audio: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("failed to create wav: {}", path.display()))?;

    for &x in audio {
        let y = x.clamp(-1.0, 1.0);
        let s = (y * i16::MAX as f32).round() as i16;
        writer
            .write_sample(s)
            .with_context(|| format!("failed writing sample to {}", path.display()))?;
    }
    writer
        .finalize()
        .with_context(|| format!("failed finalizing wav: {}", path.display()))?;
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let in_wav = match args.next() {
        Some(p) => PathBuf::from(p),
        None => {
            bail!(
                "usage: cargo run --release --bin infer_wav -- <input.wav> [output.wav] [model_dir]"
            )
        }
    };

    let out_wav = args.next().map(PathBuf::from).unwrap_or_else(|| {
        in_wav.with_file_name(format!(
            "{}_rust_enhanced.wav",
            in_wav
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("output")
        ))
    });

    let model_dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../checkpoints/dfn_style"));

    let (audio, sr) = read_wav_mono_f32(&in_wav)?;
    let orig_len = audio.len();

    let mut padded = audio;
    let rem = padded.len() % HOP_SIZE;
    if rem != 0 {
        padded.resize(padded.len() + (HOP_SIZE - rem), 0.0);
    }

    let mut processor = LightDFNProcessor::new(&model_dir)
        .with_context(|| format!("failed to init processor from {}", model_dir.display()))?;

    let mut out = Vec::with_capacity(padded.len());
    for hop in padded.chunks_exact(HOP_SIZE) {
        let y = processor.process_frame(hop)?;
        out.extend_from_slice(&y);
    }

    out.truncate(orig_len);

    let peak = out
        .iter()
        .fold(0.0f32, |m, &v| if v.abs() > m { v.abs() } else { m });
    if peak > 1.0 {
        for s in &mut out {
            *s /= peak;
        }
    }

    write_wav_mono_i16(&out_wav, sr, &out)?;
    println!("Saved enhanced audio -> {}", out_wav.display());
    Ok(())
}
