//! Serde JSON and Bincode model and optimizer checkpointing.

use crate::error::{EngineError, Result};
use crate::tensor::RawTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Training checkpoint containing epoch and named tensor buffers.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Checkpoint {
    pub epoch: usize,
    pub step: usize,
    pub tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Checkpoint {
    pub fn new(epoch: usize, step: usize) -> Self {
        Self {
            epoch,
            step,
            tensors: HashMap::new(),
        }
    }

    pub fn insert(&mut self, name: &str, tensor: &RawTensor) {
        let contig = tensor.to_contiguous();
        self.tensors.insert(
            name.to_string(),
            (contig.shape().to_vec(), contig.as_slice().to_vec()),
        );
    }

    pub fn get(&self, name: &str) -> Option<RawTensor> {
        self.tensors
            .get(name)
            .map(|(shape, data)| RawTensor::from_vec(data.clone(), shape.clone()))
    }

    /// Saves the checkpoint to a JSON file.
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(|e| {
            EngineError::SerializationError(format!("Failed to serialize JSON checkpoint: {}", e))
        })?;
        let mut file = File::create(path).map_err(|e| {
            EngineError::SerializationError(format!("Failed to create file: {}", e))
        })?;
        file.write_all(json.as_bytes())
            .map_err(|e| EngineError::SerializationError(format!("Failed to write file: {}", e)))?;
        Ok(())
    }

    /// Loads a checkpoint from a JSON file.
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path)
            .map_err(|e| EngineError::SerializationError(format!("Failed to open file: {}", e)))?;
        let mut text = String::new();
        file.read_to_string(&mut text)
            .map_err(|e| EngineError::SerializationError(format!("Failed to read file: {}", e)))?;
        let cp: Self = serde_json::from_str(&text).map_err(|e| {
            EngineError::SerializationError(format!("Failed to parse JSON checkpoint: {}", e))
        })?;
        Ok(cp)
    }

    /// Saves the checkpoint to a compact binary bincode file.
    pub fn save_bincode<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let bytes = bincode::serialize(self).map_err(|e| {
            EngineError::SerializationError(format!(
                "Failed to serialize bincode checkpoint: {}",
                e
            ))
        })?;
        let mut file = File::create(path).map_err(|e| {
            EngineError::SerializationError(format!("Failed to create file: {}", e))
        })?;
        file.write_all(&bytes)
            .map_err(|e| EngineError::SerializationError(format!("Failed to write file: {}", e)))?;
        Ok(())
    }

    /// Loads a checkpoint from a binary bincode file.
    pub fn load_bincode<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path)
            .map_err(|e| EngineError::SerializationError(format!("Failed to open file: {}", e)))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| EngineError::SerializationError(format!("Failed to read file: {}", e)))?;
        let cp: Self = bincode::deserialize(&buffer).map_err(|e| {
            EngineError::SerializationError(format!("Failed to parse bincode checkpoint: {}", e))
        })?;
        Ok(cp)
    }
}
