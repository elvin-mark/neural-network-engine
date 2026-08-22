//! Audio processing, Short-Time Fourier Transform (STFT), Mel filterbanks,
//! and spoken audio speech dataset generation.

use crate::tensor::RawTensor;
use rand::Rng;
use std::f32::consts::PI;

/// Converts frequency in Hertz to Mel scale.
#[inline]
pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Converts Mel scale value back to frequency in Hertz.
#[inline]
pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

/// Constructs a triangular Mel filterbank matrix of shape `[n_mels, n_fft / 2 + 1]`.
pub fn create_mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: usize,
    f_min: f32,
    f_max: f32,
) -> Vec<Vec<f32>> {
    let num_bins = n_fft / 2 + 1;
    let min_mel = hz_to_mel(f_min);
    let max_mel = hz_to_mel(f_max);

    let mut mel_points = Vec::with_capacity(n_mels + 2);
    for i in 0..=(n_mels + 1) {
        let mel = min_mel + (max_mel - min_mel) * (i as f32) / ((n_mels + 1) as f32);
        mel_points.push(mel_to_hz(mel));
    }

    let mut bin_points = Vec::with_capacity(n_mels + 2);
    for &hz in &mel_points {
        let bin = ((n_fft as f32 + 1.0) * hz / (sample_rate as f32)).floor() as usize;
        bin_points.push(bin.min(num_bins - 1));
    }

    let mut filterbank = vec![vec![0.0f32; num_bins]; n_mels];

    for (m, row) in filterbank.iter_mut().enumerate().take(n_mels) {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        if center > left {
            for (k, val) in row.iter_mut().enumerate().take(center).skip(left) {
                *val = (k - left) as f32 / (center - left) as f32;
            }
        }
        if right > center {
            for (k, val) in row
                .iter_mut()
                .enumerate()
                .take((right + 1).min(num_bins))
                .skip(center)
            {
                *val = (right - k) as f32 / (right - center) as f32;
            }
        }
    }

    filterbank
}

/// Computes a discrete Fourier transform magnitude spectrum for a single windowed frame.
fn compute_frame_spectrum(frame: &[f32], n_fft: usize) -> Vec<f32> {
    let num_bins = n_fft / 2 + 1;
    let mut magnitudes = Vec::with_capacity(num_bins);

    for k in 0..num_bins {
        let mut real = 0.0f32;
        let mut imag = 0.0f32;
        let angle_step = -2.0 * PI * (k as f32) / (n_fft as f32);

        for (n, &val) in frame.iter().enumerate() {
            let angle = angle_step * (n as f32);
            real += val * angle.cos();
            imag += val * angle.sin();
        }

        magnitudes.push((real * real + imag * imag).sqrt());
    }

    magnitudes
}

/// Computes a Log-Mel Spectrogram of shape `[n_mels, num_frames]` from a 1D raw audio waveform.
pub fn compute_log_mel_spectrogram(
    waveform: &[f32],
    sample_rate: usize,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
) -> RawTensor {
    if waveform.len() < n_fft {
        // Zero-pad if waveform is shorter than one FFT window
        let mut padded = waveform.to_vec();
        padded.resize(n_fft, 0.0);
        return compute_log_mel_spectrogram(&padded, sample_rate, n_fft, hop_length, n_mels);
    }

    let num_frames = (waveform.len() - n_fft) / hop_length + 1;
    let filterbank =
        create_mel_filterbank(n_mels, n_fft, sample_rate, 0.0, (sample_rate / 2) as f32);

    // Pre-calculate Hann window
    let mut window = Vec::with_capacity(n_fft);
    for n in 0..n_fft {
        window.push(0.5 * (1.0 - (2.0 * PI * (n as f32) / (n_fft as f32)).cos()));
    }

    let mut mel_spectrogram = vec![0.0f32; n_mels * num_frames];

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_length;
        let mut windowed = Vec::with_capacity(n_fft);
        for i in 0..n_fft {
            windowed.push(waveform[start + i] * window[i]);
        }

        let spectrum = compute_frame_spectrum(&windowed, n_fft);

        for mel_idx in 0..n_mels {
            let mut mel_energy = 0.0f32;
            for (bin, &mag) in spectrum.iter().enumerate() {
                mel_energy += mag * filterbank[mel_idx][bin];
            }
            // Log-mel energy with numerical flooring
            let log_energy = (mel_energy.max(1e-5)).ln();
            mel_spectrogram[mel_idx * num_frames + frame_idx] = log_energy;
        }
    }

    RawTensor::from_vec(mel_spectrogram, vec![n_mels, num_frames])
}

/// Spoken word vocabulary classes.
pub const SPOKEN_CLASSES: &[&str] = &[
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "start",
    "stop", "yes", "no",
];

/// Formant frequency presets (F1, F2, F3 in Hz) for acoustic speech synthesis.
const SPOKEN_FORMANTS: &[(f32, f32, f32)] = &[
    (400.0, 1800.0, 2600.0), // zero
    (500.0, 1000.0, 2400.0), // one
    (300.0, 900.0, 2200.0),  // two
    (450.0, 1700.0, 2500.0), // three
    (550.0, 850.0, 2300.0),  // four
    (650.0, 1400.0, 2450.0), // five
    (380.0, 1950.0, 2700.0), // six
    (520.0, 1750.0, 2550.0), // seven
    (500.0, 1850.0, 2600.0), // eight
    (600.0, 1500.0, 2500.0), // nine
    (700.0, 1200.0, 2400.0), // start
    (550.0, 950.0, 2350.0),  // stop
    (350.0, 1900.0, 2650.0), // yes
    (480.0, 900.0, 2250.0),  // no
];

/// Synthesizes a realistic audio waveform for a spoken word using acoustic formant synthesis.
pub fn synthesize_spoken_word(word_idx: usize, duration_secs: f32, sample_rate: usize) -> Vec<f32> {
    let (f1, f2, f3) = SPOKEN_FORMANTS[word_idx % SPOKEN_FORMANTS.len()];
    let total_samples = (duration_secs * sample_rate as f32) as usize;
    let mut waveform = Vec::with_capacity(total_samples);

    let mut rng = rand::thread_rng();
    let f0: f32 = 120.0 + rng.gen_range(-15.0..15.0); // Pitch variation

    for t in 0..total_samples {
        let time = (t as f32) / (sample_rate as f32);
        // Amplitude envelope: attack, sustain, decay
        let env = ((PI * time / duration_secs).sin()).powf(1.5);

        // Harmonic glottal pulse excitation with formants
        let mut sample = 0.0f32;
        let num_harmonics = 15;
        for h in 1..=num_harmonics {
            let freq = (h as f32) * f0;
            // Resonance boosts near formants
            let res1 = 1.0 / (1.0 + ((freq - f1) / 80.0).powi(2));
            let res2 = 0.7 / (1.0 + ((freq - f2) / 100.0).powi(2));
            let res3 = 0.4 / (1.0 + ((freq - f3) / 120.0).powi(2));
            let weight = (1.0 / (h as f32)) * (1.0 + 3.0 * res1 + 2.0 * res2 + 1.5 * res3);

            sample += weight * (2.0 * PI * freq * time).sin();
        }

        // Add subtle background breath noise
        let noise: f32 = rng.gen_range(-0.02..0.02);
        waveform.push(env * (sample * 0.1 + noise));
    }

    waveform
}

/// Generates a batch of spoken audio Log-Mel Spectrograms of shape `[num_samples, n_mels, time_steps]`
/// paired with text transcription labels.
pub fn generate_spoken_dataset(
    num_samples: usize,
    n_mels: usize,
    time_steps: usize,
) -> (RawTensor, Vec<String>) {
    let sample_rate = 8000;
    let n_fft = 128;
    let hop_length = 32;
    let duration = (time_steps * hop_length + n_fft) as f32 / (sample_rate as f32);

    let mut all_specs = Vec::with_capacity(num_samples * n_mels * time_steps);
    let mut labels = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let word_idx = i % SPOKEN_CLASSES.len();
        let label = SPOKEN_CLASSES[word_idx].to_string();

        let waveform = synthesize_spoken_word(word_idx, duration, sample_rate);
        let spec = compute_log_mel_spectrogram(&waveform, sample_rate, n_fft, hop_length, n_mels);
        let spec_data = spec.as_slice();
        let num_frames = spec.shape()[1];

        // Crop or pad to exact time_steps
        for m in 0..n_mels {
            for t in 0..time_steps {
                if t < num_frames {
                    all_specs.push(spec_data[m * num_frames + t]);
                } else {
                    all_specs.push(-11.51); // floor value log(1e-5)
                }
            }
        }

        labels.push(label);
    }

    (
        RawTensor::from_vec(all_specs, vec![num_samples, n_mels, time_steps]),
        labels,
    )
}

/// Loads or generates the spoken speech dataset.
pub fn load_spoken_dataset(num_samples: Option<usize>) -> (RawTensor, Vec<String>) {
    let n = num_samples.unwrap_or(280);
    generate_spoken_dataset(n, 64, 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_filterbank_generation() {
        let fb = create_mel_filterbank(40, 256, 16000, 0.0, 8000.0);
        assert_eq!(fb.len(), 40);
        assert_eq!(fb[0].len(), 129);
        for filter in &fb {
            assert!(filter.iter().any(|&v| v > 0.0));
        }
    }

    #[test]
    fn test_log_mel_spectrogram_computation() {
        let mut waveform = vec![0.0f32; 1600];
        for (i, v) in waveform.iter_mut().enumerate() {
            *v = (2.0 * PI * 440.0 * (i as f32) / 8000.0).sin();
        }

        let spec = compute_log_mel_spectrogram(&waveform, 8000, 128, 32, 32);
        assert_eq!(spec.ndim(), 2);
        assert_eq!(spec.shape()[0], 32);
        assert!(spec.shape()[1] > 10);
    }

    #[test]
    fn test_spoken_dataset_generation() {
        let (specs, labels) = generate_spoken_dataset(14, 64, 32);
        assert_eq!(specs.shape(), &[14, 64, 32]);
        assert_eq!(labels.len(), 14);
        assert_eq!(labels[0], "zero");
        assert_eq!(labels[1], "one");
    }
}
