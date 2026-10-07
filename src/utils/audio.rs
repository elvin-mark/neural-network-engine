//! Audio processing, Short-Time Fourier Transform (STFT), Mel filterbanks,
//! and spoken audio speech dataset generation.

use crate::tensor::RawTensor;
use rand::Rng;
use rayon::prelude::*;
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

/// Slaney-style Hertz to Mel conversion (linear below 1000 Hz, logarithmic above).
fn slaney_hz_to_mel(hz: f32) -> f32 {
    let min_log_hertz = 1000.0f32;
    let min_log_mel = 15.0f32;
    let logstep = 27.0f32 / 6.4f32.ln();

    if hz < min_log_hertz {
        3.0 * hz / 200.0
    } else {
        min_log_mel + (hz / min_log_hertz).ln() * logstep
    }
}

/// Slaney-style Mel to Hertz conversion.
fn slaney_mel_to_hz(mel: f32) -> f32 {
    let min_log_hertz = 1000.0f32;
    let min_log_mel = 15.0f32;
    let logstep = 6.4f32.ln() / 27.0f32;

    if mel < min_log_mel {
        200.0 * mel / 3.0
    } else {
        min_log_hertz * (logstep * (mel - min_log_mel)).exp()
    }
}

/// Constructs the exact 80-channel Slaney triangular Mel filterbank matrix of shape `[201, 80]`
/// matching OpenAI Whisper / HuggingFace `WhisperFeatureExtractor`.
pub fn create_whisper_mel_filterbank() -> Vec<Vec<f32>> {
    let num_mel_filters = 80;
    let num_freq_bins = 201; // 400 // 2 + 1
    let min_freq = 0.0f32;
    let max_freq = 8000.0f32;

    let mel_min = slaney_hz_to_mel(min_freq);
    let mel_max = slaney_hz_to_mel(max_freq);

    let mut filter_freqs = Vec::with_capacity(num_mel_filters + 2);
    for i in 0..(num_mel_filters + 2) {
        let mel = mel_min + (mel_max - mel_min) * (i as f32) / ((num_mel_filters + 1) as f32);
        filter_freqs.push(slaney_mel_to_hz(mel));
    }

    let mut fft_freqs = Vec::with_capacity(num_freq_bins);
    for k in 0..num_freq_bins {
        fft_freqs.push((k as f32) * max_freq / ((num_freq_bins - 1) as f32));
    }

    let mut filter_diff = Vec::with_capacity(num_mel_filters + 1);
    for i in 0..(num_mel_filters + 1) {
        filter_diff.push(filter_freqs[i + 1] - filter_freqs[i]);
    }

    // mel_filters: [num_freq_bins, num_mel_filters]
    let mut mel_filters = vec![vec![0.0f32; num_mel_filters]; num_freq_bins];

    for i in 0..num_mel_filters {
        let f_diff_down = filter_diff[i];
        let f_diff_up = filter_diff[i + 1];
        let enorm = 2.0 / (filter_freqs[i + 2] - filter_freqs[i]);

        for (k, &f) in fft_freqs.iter().enumerate().take(num_freq_bins) {
            let slope_down = -(filter_freqs[i] - f) / f_diff_down;
            let slope_up = (filter_freqs[i + 2] - f) / f_diff_up;
            let val = slope_down.min(slope_up).max(0.0);
            mel_filters[k][i] = val * enorm;
        }
    }

    mel_filters
}

/// Computes the official 80-channel Log-Mel Spectrogram padded to exactly 3000 frames (30 seconds)
/// matching OpenAI Whisper's acoustic frontend and `WhisperFeatureExtractor`.
pub fn compute_whisper_mel_spectrogram(audio: &[f32]) -> RawTensor {
    let target_len = 480_000; // 30 seconds at 16,000 Hz
    let n_fft = 400;
    let hop_length = 160;
    let num_mel_bins = 80;
    let whisper_frames = 3000;

    // 1. Pad or truncate audio to 30 seconds (480,000 samples)
    let mut audio_padded = Vec::with_capacity(target_len);
    if audio.len() < target_len {
        audio_padded.extend_from_slice(audio);
        audio_padded.resize(target_len, 0.0);
    } else {
        audio_padded.extend_from_slice(&audio[..target_len]);
    }

    // 2. Reflect padding of n_fft / 2 = 200 on both sides (center = True)
    let pad_amount = n_fft / 2;
    let mut audio_centered = Vec::with_capacity(audio_padded.len() + 2 * pad_amount);

    // Left reflect padding (e.g. indices 200, 199, ..., 1)
    for p in (1..=pad_amount).rev() {
        audio_centered.push(audio_padded[p]);
    }
    audio_centered.extend_from_slice(&audio_padded);
    // Right reflect padding
    let last_idx = audio_padded.len() - 1;
    for p in 1..=pad_amount {
        audio_centered.push(audio_padded[last_idx.saturating_sub(p)]);
    }

    // 3. Pre-compute periodic Hann window: 0.5 - 0.5 * cos(2 * pi * n / N)
    let mut window = Vec::with_capacity(n_fft);
    for n in 0..n_fft {
        window.push(0.5 * (1.0 - (2.0 * PI * (n as f32) / (n_fft as f32)).cos()));
    }

    // 4. Compute STFT power spectrogram
    let filterbank = create_whisper_mel_filterbank();

    let frame_mels: Vec<Vec<f32>> = (0..whisper_frames)
        .into_par_iter()
        .map(|frame_idx| {
            let start = frame_idx * hop_length;
            let mut windowed = vec![0.0f32; n_fft];
            for i in 0..n_fft {
                windowed[i] = audio_centered[start + i] * window[i];
            }

            let spectrum = compute_frame_spectrum(&windowed, n_fft); // 201 bins
            let mut mels = vec![0.0f32; num_mel_bins];
            for m in 0..num_mel_bins {
                let mut energy = 0.0f32;
                for bin in 0..201 {
                    let p = spectrum[bin] * spectrum[bin];
                    energy += p * filterbank[bin][m];
                }
                mels[m] = energy.max(1e-10).log10();
            }
            mels
        })
        .collect();

    let mut mel_out = vec![0.0f32; num_mel_bins * whisper_frames];
    for frame_idx in 0..whisper_frames {
        for m in 0..num_mel_bins {
            mel_out[m * whisper_frames + frame_idx] = frame_mels[frame_idx][m];
        }
    }

    // 5. Dynamic range normalization:
    // log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    // log_spec = (log_spec + 4.0) / 4.0
    let mut max_val = f32::NEG_INFINITY;
    for &v in &mel_out {
        if v > max_val {
            max_val = v;
        }
    }

    let floor_val = max_val - 8.0;
    for v in &mut mel_out {
        let clamped = v.max(floor_val);
        *v = (clamped + 4.0) / 4.0;
    }

    RawTensor::from_vec(mel_out, vec![num_mel_bins, whisper_frames])
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

/// Decoded WAV audio container with normalized mono samples in range `[-1.0, 1.0]`.
#[derive(Debug, Clone)]
pub struct WavAudio {
    /// Original audio sample rate in Hz (e.g. 16000, 44100, 48000).
    pub sample_rate: u32,
    /// Number of audio channels in the source file.
    pub channels: u16,
    /// Bit depth per sample (e.g. 8, 16, 24, 32).
    pub bits_per_sample: u16,
    /// Normalized audio waveform downmixed to mono (`f32` in `[-1.0, 1.0]`).
    pub samples: Vec<f32>,
}

impl WavAudio {
    /// Returns audio duration in seconds.
    pub fn duration_seconds(&self) -> f32 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.samples.len() as f32 / self.sample_rate as f32
        }
    }

    /// Resamples the internal audio buffer to a target sample rate (e.g. 16000 Hz) in-place.
    pub fn resample(&mut self, target_sample_rate: u32) {
        if self.sample_rate == target_sample_rate || self.samples.is_empty() {
            return;
        }
        self.samples = resample_linear(&self.samples, self.sample_rate, target_sample_rate);
        self.sample_rate = target_sample_rate;
    }
}

/// Resamples a 1D audio waveform from `src_rate` to `target_rate` using linear interpolation.
pub fn resample_linear(samples: &[f32], src_rate: u32, target_rate: u32) -> Vec<f32> {
    if src_rate == target_rate || samples.is_empty() {
        return samples.to_vec();
    }

    let ratio = src_rate as f64 / target_rate as f64;
    let target_len = ((samples.len() as f64) / ratio).round() as usize;
    let mut output = Vec::with_capacity(target_len);

    for i in 0..target_len {
        let src_idx = (i as f64) * ratio;
        let idx0 = src_idx.floor() as usize;
        let idx1 = (idx0 + 1).min(samples.len() - 1);
        let frac = (src_idx - idx0 as f64) as f32;

        if idx0 < samples.len() {
            let s0 = samples[idx0];
            let s1 = samples[idx1];
            output.push(s0 + frac * (s1 - s0));
        }
    }

    output
}

/// Reads a standard RIFF/WAVE audio file (.wav) without any external dependencies.
/// Supports 8-bit unsigned PCM, 16-bit signed PCM, 24-bit signed PCM, 32-bit signed PCM,
/// and 32-bit IEEE float. Automatically downmixes multi-channel audio to mono.
pub fn read_wav_file<P: AsRef<std::path::Path>>(path: P) -> crate::error::Result<WavAudio> {
    use crate::error::EngineError;
    use std::fs::File;
    use std::io::Read;

    let path_ref = path.as_ref();
    let mut file = File::open(path_ref).map_err(|e| {
        EngineError::SerializationError(format!("Failed to open WAV file {:?}: {}", path_ref, e))
    })?;

    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).map_err(|e| {
        EngineError::SerializationError(format!("Failed to read WAV file {:?}: {}", path_ref, e))
    })?;

    parse_wav_bytes(&bytes)
}

/// Parses raw bytes of a RIFF/WAVE container.
pub fn parse_wav_bytes(bytes: &[u8]) -> crate::error::Result<WavAudio> {
    use crate::error::EngineError;

    if bytes.len() < 44 {
        return Err(EngineError::SerializationError(
            "WAV file is too small to contain valid RIFF headers".to_string(),
        ));
    }

    // 1. Verify "RIFF" and "WAVE"
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(EngineError::SerializationError(
            "Invalid WAV file format: missing RIFF/WAVE signature".to_string(),
        ));
    }

    let mut cursor = 12;
    let mut audio_format: Option<u16> = None;
    let mut channels: Option<u16> = None;
    let mut sample_rate: Option<u32> = None;
    let mut bits_per_sample: Option<u16> = None;
    let mut data_chunk_range: Option<(usize, usize)> = None;

    while cursor + 8 <= bytes.len() {
        let chunk_id = &bytes[cursor..cursor + 4];
        let chunk_size = u32::from_le_bytes([
            bytes[cursor + 4],
            bytes[cursor + 5],
            bytes[cursor + 6],
            bytes[cursor + 7],
        ]) as usize;
        cursor += 8;

        if chunk_id == b"fmt " {
            if chunk_size < 16 || cursor + 16 > bytes.len() {
                return Err(EngineError::SerializationError(
                    "Malformed 'fmt ' chunk in WAV header".to_string(),
                ));
            }
            audio_format = Some(u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]));
            channels = Some(u16::from_le_bytes([bytes[cursor + 2], bytes[cursor + 3]]));
            sample_rate = Some(u32::from_le_bytes([
                bytes[cursor + 4],
                bytes[cursor + 5],
                bytes[cursor + 6],
                bytes[cursor + 7],
            ]));
            bits_per_sample = Some(u16::from_le_bytes([bytes[cursor + 14], bytes[cursor + 15]]));
        } else if chunk_id == b"data" {
            let data_end = (cursor + chunk_size).min(bytes.len());
            data_chunk_range = Some((cursor, data_end));
            break; // Standard primary data chunk found
        }

        // Advance to next chunk (chunks are 2-byte aligned in RIFF)
        cursor += chunk_size;
        if chunk_size % 2 != 0 {
            cursor += 1;
        }
    }

    let fmt = audio_format.ok_or_else(|| {
        EngineError::SerializationError("Missing 'fmt ' chunk in WAV file".to_string())
    })?;
    let ch = channels.ok_or_else(|| {
        EngineError::SerializationError("Missing channels in WAV file".to_string())
    })?;
    let sr = sample_rate.ok_or_else(|| {
        EngineError::SerializationError("Missing sample rate in WAV file".to_string())
    })?;
    let bits = bits_per_sample.ok_or_else(|| {
        EngineError::SerializationError("Missing bits per sample in WAV file".to_string())
    })?;
    let (data_start, data_end) = data_chunk_range.ok_or_else(|| {
        EngineError::SerializationError("Missing 'data' chunk in WAV file".to_string())
    })?;

    if ch == 0 {
        return Err(EngineError::SerializationError(
            "Invalid channel count (0)".to_string(),
        ));
    }

    let raw_data = &bytes[data_start..data_end];
    let num_channels = ch as usize;

    let samples: Vec<f32> = match (fmt, bits) {
        // PCM 8-bit unsigned
        (1, 8) => {
            let total_samples = raw_data.len();
            let num_frames = total_samples / num_channels;
            let mut mono = Vec::with_capacity(num_frames);
            for frame_idx in 0..num_frames {
                let mut sum = 0.0f32;
                for c in 0..num_channels {
                    let b = raw_data[frame_idx * num_channels + c] as f32;
                    sum += (b - 128.0) / 128.0;
                }
                mono.push(sum / (num_channels as f32));
            }
            mono
        }
        // PCM 16-bit signed
        (1, 16) => {
            let total_samples = raw_data.len() / 2;
            let num_frames = total_samples / num_channels;
            let mut mono = Vec::with_capacity(num_frames);
            for frame_idx in 0..num_frames {
                let mut sum = 0.0f32;
                for c in 0..num_channels {
                    let offset = (frame_idx * num_channels + c) * 2;
                    if offset + 2 <= raw_data.len() {
                        let sample_i16 =
                            i16::from_le_bytes([raw_data[offset], raw_data[offset + 1]]) as f32;
                        sum += sample_i16 / 32768.0;
                    }
                }
                mono.push(sum / (num_channels as f32));
            }
            mono
        }
        // PCM 24-bit signed
        (1, 24) => {
            let total_samples = raw_data.len() / 3;
            let num_frames = total_samples / num_channels;
            let mut mono = Vec::with_capacity(num_frames);
            for frame_idx in 0..num_frames {
                let mut sum = 0.0f32;
                for c in 0..num_channels {
                    let offset = (frame_idx * num_channels + c) * 3;
                    if offset + 3 <= raw_data.len() {
                        let b0 = raw_data[offset] as u32;
                        let b1 = raw_data[offset + 1] as u32;
                        let b2 = raw_data[offset + 2] as u32;
                        let raw_24 = b0 | (b1 << 8) | (b2 << 16);
                        // Sign extend 24-bit to 32-bit
                        let sample_i32 = if (raw_24 & 0x800000) != 0 {
                            (raw_24 | 0xFF000000) as i32
                        } else {
                            raw_24 as i32
                        } as f32;
                        sum += sample_i32 / 8388608.0;
                    }
                }
                mono.push(sum / (num_channels as f32));
            }
            mono
        }
        // PCM 32-bit signed
        (1, 32) => {
            let total_samples = raw_data.len() / 4;
            let num_frames = total_samples / num_channels;
            let mut mono = Vec::with_capacity(num_frames);
            for frame_idx in 0..num_frames {
                let mut sum = 0.0f32;
                for c in 0..num_channels {
                    let offset = (frame_idx * num_channels + c) * 4;
                    if offset + 4 <= raw_data.len() {
                        let sample_i32 = i32::from_le_bytes([
                            raw_data[offset],
                            raw_data[offset + 1],
                            raw_data[offset + 2],
                            raw_data[offset + 3],
                        ]) as f32;
                        sum += sample_i32 / 2147483648.0;
                    }
                }
                mono.push(sum / (num_channels as f32));
            }
            mono
        }
        // IEEE Float 32-bit
        (3, 32) => {
            let total_samples = raw_data.len() / 4;
            let num_frames = total_samples / num_channels;
            let mut mono = Vec::with_capacity(num_frames);
            for frame_idx in 0..num_frames {
                let mut sum = 0.0f32;
                for c in 0..num_channels {
                    let offset = (frame_idx * num_channels + c) * 4;
                    if offset + 4 <= raw_data.len() {
                        let val = f32::from_le_bytes([
                            raw_data[offset],
                            raw_data[offset + 1],
                            raw_data[offset + 2],
                            raw_data[offset + 3],
                        ]);
                        sum += val;
                    }
                }
                mono.push(sum / (num_channels as f32));
            }
            mono
        }
        (other_fmt, other_bits) => {
            return Err(EngineError::SerializationError(format!(
                "Unsupported WAV format: format_code={}, bits_per_sample={}",
                other_fmt, other_bits
            )));
        }
    };

    Ok(WavAudio {
        sample_rate: sr,
        channels: ch,
        bits_per_sample: bits,
        samples,
    })
}

/// Encodes normalized float samples (`[-1.0, 1.0]`) into a 16-bit PCM mono WAV file.
pub fn write_wav_file<P: AsRef<std::path::Path>>(
    path: P,
    samples: &[f32],
    sample_rate: u32,
) -> crate::error::Result<()> {
    use crate::error::EngineError;
    use std::fs::File;
    use std::io::Write;

    let path_ref = path.as_ref();
    let mut file = File::create(path_ref).map_err(|e| {
        EngineError::SerializationError(format!("Failed to create WAV file {:?}: {}", path_ref, e))
    })?;

    let num_channels = 1u16;
    let bits_per_sample = 16u16;
    let bytes_per_sample = 2usize;
    let data_len = samples.len() * bytes_per_sample;
    let riff_chunk_size = (36 + data_len) as u32;

    let mut header = Vec::with_capacity(44);
    // RIFF Header
    header.extend_from_slice(b"RIFF");
    header.extend_from_slice(&riff_chunk_size.to_le_bytes());
    header.extend_from_slice(b"WAVE");

    // "fmt " Subchunk
    header.extend_from_slice(b"fmt ");
    header.extend_from_slice(&16u32.to_le_bytes()); // Subchunk1Size for PCM
    header.extend_from_slice(&1u16.to_le_bytes()); // AudioFormat = 1 (PCM)
    header.extend_from_slice(&num_channels.to_le_bytes());
    header.extend_from_slice(&sample_rate.to_le_bytes());
    let byte_rate = sample_rate * (num_channels as u32) * (bytes_per_sample as u32);
    header.extend_from_slice(&byte_rate.to_le_bytes());
    let block_align = num_channels * (bytes_per_sample as u16);
    header.extend_from_slice(&block_align.to_le_bytes());
    header.extend_from_slice(&bits_per_sample.to_le_bytes());

    // "data" Subchunk
    header.extend_from_slice(b"data");
    header.extend_from_slice(&(data_len as u32).to_le_bytes());

    file.write_all(&header).map_err(|e| {
        EngineError::SerializationError(format!("Failed to write WAV header: {}", e))
    })?;

    // Write 16-bit PCM samples
    let mut sample_bytes = Vec::with_capacity(data_len);
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let val_i16 = (clamped * 32767.0).round() as i16;
        sample_bytes.extend_from_slice(&val_i16.to_le_bytes());
    }

    file.write_all(&sample_bytes)
        .map_err(|e| EngineError::SerializationError(format!("Failed to write WAV data: {}", e)))?;

    Ok(())
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

    #[test]
    fn test_wav_write_read_roundtrip() {
        let sample_rate = 16000;
        let mut original_samples = Vec::new();
        for i in 0..1600 {
            // 440 Hz sine wave
            original_samples.push((2.0 * PI * 440.0 * (i as f32) / (sample_rate as f32)).sin());
        }

        let temp_path = std::env::temp_dir().join("nne_test_roundtrip.wav");
        write_wav_file(&temp_path, &original_samples, sample_rate).unwrap();

        let wav = read_wav_file(&temp_path).unwrap();
        assert_eq!(wav.sample_rate, 16000);
        assert_eq!(wav.channels, 1);
        assert_eq!(wav.bits_per_sample, 16);
        assert_eq!(wav.samples.len(), original_samples.len());

        for (a, b) in original_samples.iter().zip(wav.samples.iter()) {
            assert!((a - b).abs() < 0.001);
        }

        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_resample_linear() {
        let src_rate = 32000;
        let target_rate = 16000;
        let samples = vec![0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5];

        let resampled = resample_linear(&samples, src_rate, target_rate);
        assert_eq!(resampled.len(), samples.len() / 2);
        assert!((resampled[0] - 0.0).abs() < 1e-4);
        assert!((resampled[1] - 1.0).abs() < 1e-4);
    }
}
