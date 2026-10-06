# Audio Processing & Mel Frontend (`utils::audio`)

Pure-Rust audio signal processing pipeline for automatic speech recognition (ASR) matching OpenAI Whisper's acoustic frontend.

**Source files**:
- Core implementation: [`src/utils/audio.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/utils/audio.rs)
- Whisper Model: [`src/models/whisper.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/models/whisper.rs)
- Example: [`examples/12_whisper_speech_recognition.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/examples/12_whisper_speech_recognition.rs)

---

## The Acoustic Frontend Pipeline

```mermaid
flowchart LR
    Wave["Raw Audio Waveform (16 kHz)"] --> Hann["Hann Windowing (N_fft = 400, Hop = 160)"]
    Hann --> STFT["Short-Time Fourier Transform (STFT)"]
    STFT --> Power["Power Spectrum: |STFT|^2"]
    Power --> Filterbank["80-channel Triangular Mel Filterbank"]
    Filterbank --> Log["Log Compression: clamp(log10(x))"]
    Log --> MelOut["Log-Mel Spectrogram: [80, 3000]"]
```

---

## Mathematical Formulations

### 1. Hertz to Mel Conversion (Slaney / HTK Formula)
$$\text{mel}(f) = 2595 \times \log_{10}\left(1 + \frac{f}{700}\right)$$
$$f(\text{mel}) = 700 \times \left(10^{\frac{\text{mel}}{2595}} - 1\right)$$

### 2. Triangular Mel Filterbank
Generates $M$ triangular filters spaced uniformly along the mel scale between $0\text{ Hz}$ and $\frac{f_s}{2} = 8000\text{ Hz}$:

```rust
pub fn create_mel_filterbank(
    num_mel_bins: usize, // 80 (Whisper standard)
    n_fft: usize,        // 400
    sample_rate: usize,  // 16000 Hz
) -> Result<RawTensor>
```

### 3. Log-Mel Spectrogram Computation
Computes STFT using Hann windowing and hop length of 160 samples (10 ms frame shift), maps power spectrum through the filterbank, and applies logarithmic compression:

```rust
pub fn compute_log_mel_spectrogram(
    waveform: &[f32],
    num_mel_bins: usize,
) -> Result<RawTensor>
```

The resulting spectrogram is normalized such that silence corresponds to $-1.0$ and maximal activations to $+1.0$.

---

## See Also
- [Whisper Model Architecture](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/models/whisper.md)
- [Dataset Utilities](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/infra/data-and-vision.md)
