# Serialization & Model Checkpoints (`io`)

Serialization formats for model weights and training checkpoints: SafeTensors and binary/JSON checkpoints.

**Source files**:
- SafeTensors: [`src/io/safetensors.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/io/safetensors.rs)
- Checkpoint: [`src/io/checkpoint.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/io/checkpoint.rs)
- Conversion Script: [`scripts/convert_hf_model.py`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/scripts/convert_hf_model.py)

---

## SafeTensors Format (`io::safetensors`)

The engine provides full zero-copy serialization and deserialization for HuggingFace's standard `.safetensors` file specification.

### 1. Saving Weights
```rust
use neural_network_engine::io::save_safetensors;
use std::collections::HashMap;

let mut tensors = HashMap::new();
tensors.insert("encoder.layer.0.weight".to_string(), weight_tensor);
save_safetensors(&tensors, "model.safetensors")?;
```

### 2. Loading Weights
```rust
use neural_network_engine::io::load_safetensors;

let weights_map = load_safetensors("model.safetensors")?;
for (name, tensor) in &weights_map {
    println!("Loaded tensor '{}' with shape {:?}", name, tensor.shape());
}
```

---

## Training State Checkpointing (`io::Checkpoint`)

During model training, the `Checkpoint` struct saves and resumes:
- Model parameter tensors
- Optimizer state (momentum vectors, variance accumulators, step count)
- Epoch and iteration counters
- Best validation metric

Supports both binary format (compact, fast via `bincode`) and human-readable JSON format.

```rust
let checkpoint = Checkpoint::new(epoch, step, model_weights, optimizer_state);
checkpoint.save_binary("checkpoint_epoch_10.bin")?;
```

---

## See Also
- [Pretrained Models Catalog](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/models/README.md)
- [Optimizers](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/training/optimizers.md)
