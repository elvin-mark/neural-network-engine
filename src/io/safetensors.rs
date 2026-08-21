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
    let mut views: HashMap<String, TensorView> = HashMap::new();
    let mut raw_bytes_storage: Vec<Vec<u8>> = Vec::new();

    // Contiguous raw buffers converted to byte slices
    for tensor in tensors.values() {
        let contig = tensor.to_contiguous();
        let slice = contig.as_slice();
        let byte_len = slice.len() * 4;
        let mut byte_vec = vec![0u8; byte_len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                slice.as_ptr() as *const u8,
                byte_vec.as_mut_ptr(),
                byte_len,
            );
        }
        raw_bytes_storage.push(byte_vec);
    }

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
        let shape = view.shape().to_vec();
        let data_bytes = view.data();
        let num_floats = data_bytes.len() / 4;
        let mut float_data = vec![0.0f32; num_floats];

        unsafe {
            std::ptr::copy_nonoverlapping(
                data_bytes.as_ptr(),
                float_data.as_mut_ptr() as *mut u8,
                data_bytes.len(),
            );
        }

        let tensor = RawTensor::from_vec(float_data, shape);
        result.insert(name.to_string(), tensor);
    }

    Ok(result)
}
