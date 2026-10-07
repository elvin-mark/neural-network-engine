use neural_network_engine::prelude::*;

#[test]
fn test_whisper_encoder_decoder_forward_backward() {
    let config = WhisperConfig {
        n_mels: 32,
        d_model: 32,
        encoder_layers: 2,
        decoder_layers: 2,
        encoder_heads: 2,
        decoder_heads: 2,
        d_ff: 64,
        vocab_size: 120,
        max_source_positions: 64,
        max_target_positions: 32,
    };

    let model = Whisper::new(config);

    // Audio mel spectrogram: [Batch=2, n_mels=32, T_audio=24]
    let mel = Tensor::randn(&[2, 32, 24], 0.0, 1.0, true);
    // Target text tokens: [Batch=2, T_text=6]
    let tokens_raw = RawTensor::from_slice(
        &[
            1.0, 15.0, 22.0, 35.0, 42.0, 2.0, // Batch 0
            1.0, 18.0, 29.0, 50.0, 8.0, 2.0, // Batch 1
        ],
        &[2, 6],
    );
    let tokens = Tensor::new(tokens_raw, false);

    let logits = model.forward_model(&mel, &tokens).unwrap();
    assert_eq!(logits.shape(), &[2, 6, 120]);

    // Backward gradient propagation check
    let loss = logits.sum_all();
    loss.backward();

    assert!(mel.grad().is_some());
    assert_eq!(mel.grad().unwrap().shape(), &[2, 32, 24]);
    assert!(model.encoder.conv1.weight.grad().is_some());
    assert!(model.decoder.token_embedding.weight.grad().is_some());
    assert!(model.decoder.lm_head.weight.grad().is_some());
}

#[test]
fn test_whisper_error_handling_on_invalid_inputs() {
    let config = WhisperConfig::tiny();
    let model = Whisper::new(config);

    // Incompatible 2D mel spectrogram instead of 3D
    let invalid_mel = Tensor::randn(&[64, 32], 0.0, 1.0, false);
    let tokens = Tensor::new(RawTensor::from_slice(&[1.0, 2.0], &[1, 2]), false);

    assert!(model.forward_model(&invalid_mel, &tokens).is_err());

    // Mismatched mel bins (e.g. 32 instead of config.n_mels=64)
    let wrong_mels = Tensor::randn(&[2, 32, 30], 0.0, 1.0, false);
    assert!(model.forward_model(&wrong_mels, &tokens).is_err());
}

#[test]
fn test_whisper_audio_pipeline_and_transcription() {
    let tokenizer = ByteLevelBPE::with_special_tokens(&["<unk>", "<s>", "</s>", "<pad>"]);
    let config = WhisperConfig {
        n_mels: 32,
        d_model: 32,
        encoder_layers: 1,
        decoder_layers: 1,
        encoder_heads: 2,
        decoder_heads: 2,
        d_ff: 64,
        vocab_size: tokenizer.vocab_size(),
        max_source_positions: 64,
        max_target_positions: 32,
    };
    let model = Whisper::new(config);

    let waveform = synthesize_spoken_word(0, 0.2, 8000);
    let mel = compute_log_mel_spectrogram(&waveform, 8000, 128, 32, 32);
    let num_frames = mel.shape()[1];

    let mel_batch = mel.reshape(&[1, 32, num_frames]).unwrap();
    let mel_tensor = Tensor::new(mel_batch, false);

    let transcribed = model.generate_transcription(&mel_tensor, &tokenizer, 5);
    assert!(transcribed.is_ok());
}
