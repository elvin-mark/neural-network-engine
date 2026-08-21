//! SafeTensors format serializer and deserializer for tensor weights.

use crate::error::{EngineError, Result};
use crate::tensor::RawTensor;
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Saves a collection of named tensors to a SafeTensors file.
pub fn save_safetensors<P: AsRef<Path>>(
    tensors: &HashMap<String, RawTensor>,
    path: P,
) -> Result<()> {
    let mut raw_bytes_storage: Vec<Vec<u8>> = Vec::new();

    // Convert contiguous float buffers to little-endian byte vectors safely
    for tensor in tensors.values() {
        let contig = tensor.to_contiguous();
        let slice = contig.as_slice();
        let mut byte_vec = Vec::with_capacity(slice.len() * 4);
        for &val in slice {
            byte_vec.extend_from_slice(&val.to_le_bytes());
        }
        raw_bytes_storage.push(byte_vec);
    }

    let mut views: HashMap<String, TensorView> = HashMap::new();
    for (idx, (name, tensor)) in tensors.iter().enumerate() {
        let byte_slice = &raw_bytes_storage[idx];
        let shape = tensor.shape().to_vec();
        let view = TensorView::new(Dtype::F32, shape, byte_slice).map_err(|e| {
            EngineError::SerializationError(format!("Failed to create SafeTensors view: {}", e))
        })?;
        views.insert(name.clone(), view);
    }

    let serialized_bytes = safetensors::serialize(&views, &None).map_err(|e| {
        EngineError::SerializationError(format!("SafeTensors serialization error: {}", e))
    })?;

    let mut file = File::create(path)
        .map_err(|e| EngineError::SerializationError(format!("Failed to create file: {}", e)))?;
    file.write_all(&serialized_bytes).map_err(|e| {
        EngineError::SerializationError(format!("Failed to write SafeTensors file: {}", e))
    })?;

    Ok(())
}

/// Loads named tensors from a SafeTensors file.
pub fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<HashMap<String, RawTensor>> {
    let mut file = File::open(path).map_err(|e| {
        EngineError::SerializationError(format!("Failed to open SafeTensors file: {}", e))
    })?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).map_err(|e| {
        EngineError::SerializationError(format!("Failed to read SafeTensors file: {}", e))
    })?;

    let st = SafeTensors::deserialize(&buffer).map_err(|e| {
        EngineError::SerializationError(format!("SafeTensors deserialization error: {}", e))
    })?;

    let mut result = HashMap::new();
    for (name, view) in st.tensors() {
        if view.dtype() != Dtype::F32 {
            return Err(EngineError::SerializationError(format!(
                "Unsupported SafeTensors dtype {:?} for tensor '{}'. Only F32 is currently supported.",
                view.dtype(),
                name
            )));
        }

        let shape = view.shape().to_vec();
        let data_bytes = view.data();
        let expected_elements: usize = shape.iter().product();
        if data_bytes.len() != expected_elements * 4 {
            return Err(EngineError::SerializationError(format!(
                "Byte length mismatch for tensor '{}': expected {} bytes for shape {:?}, found {} bytes",
                name,
                expected_elements * 4,
                shape,
                data_bytes.len()
            )));
        }

        let mut float_data = Vec::with_capacity(expected_elements);
        for chunk in data_bytes.chunks_exact(4) {
            let bytes: [u8; 4] = chunk.try_into().map_err(|_| {
                EngineError::SerializationError("Failed to convert 4 bytes to f32".to_string())
            })?;
            float_data.push(f32::from_le_bytes(bytes));
        }

        let tensor = RawTensor::from_vec(float_data, shape);
        result.insert(name.to_string(), tensor);
    }

    Ok(result)
}
