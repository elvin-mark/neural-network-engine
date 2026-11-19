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
        let shape = view.shape().to_vec();
        let data_bytes = view.data();
        let expected_elements: usize = shape.iter().product();

        let elem_size = match view.dtype() {
            Dtype::F32 => 4,
            Dtype::BF16 | Dtype::F16 => 2,
            other => {
                return Err(EngineError::SerializationError(format!(
                    "Unsupported SafeTensors dtype {:?} for tensor '{}'. Supported dtypes: F32, BF16, F16.",
                    other, name
                )));
            }
        };

        if data_bytes.len() != expected_elements * elem_size {
            return Err(EngineError::SerializationError(format!(
                "Byte length mismatch for tensor '{}': expected {} bytes for shape {:?}, found {} bytes",
                name,
                expected_elements * elem_size,
                shape,
                data_bytes.len()
            )));
        }

        let mut float_data = Vec::with_capacity(expected_elements);
        match view.dtype() {
            Dtype::F32 => {
                for chunk in data_bytes.chunks_exact(4) {
                    let bytes: [u8; 4] = chunk.try_into().map_err(|_| {
                        EngineError::SerializationError(
                            "Failed to convert 4 bytes to f32".to_string(),
                        )
                    })?;
                    float_data.push(f32::from_le_bytes(bytes));
                }
            }
            Dtype::BF16 => {
                for chunk in data_bytes.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    float_data.push(f32::from_bits((bits as u32) << 16));
                }
            }
            Dtype::F16 => {
                for chunk in data_bytes.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    float_data.push(f16_to_f32(bits));
                }
            }
            _ => unreachable!(),
        }

        let tensor = RawTensor::from_vec(float_data, shape);
        result.insert(name.to_string(), tensor);
    }

    Ok(result)
}

/// Converts an IEEE 754 binary16 (f16) bit pattern to `f32`.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits & 0x7C00) >> 10) as u32;
    let frac = (bits & 0x03FF) as u32;

    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign)
        } else {
            // Subnormal
            let mut f = frac;
            let mut e = 127 - 14;
            while (f & 0x0400) == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x03FF;
            f32::from_bits(sign | (e << 23) | (f << 13))
        }
    } else if exp == 0x1F {
        // Inf or NaN
        f32::from_bits(sign | 0x7F80_0000 | (frac << 13))
    } else {
        f32::from_bits(sign | ((exp + (127 - 15)) << 23) | (frac << 13))
    }
}
