//! Shape and stride management with broadcasting for N-dimensional tensors.

use crate::error::{EngineError, Result};

/// Computes standard row-major (C-contiguous) strides for a given shape.
pub fn compute_c_contiguous_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![0; shape.len()];
    let mut current_stride = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = current_stride;
        current_stride *= shape[i].max(1);
    }
    strides
}

/// Checks whether a shape and strides combination represents a contiguous memory layout.
pub fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    if shape.is_empty() {
        return true;
    }
    let expected = compute_c_contiguous_strides(shape);
    for (i, (&dim, &stride)) in shape.iter().zip(strides.iter()).enumerate() {
        if dim > 1 && stride != expected[i] {
            return false;
        }
    }
    true
}

/// Total number of elements represented by a shape.
pub fn numel(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    }
}

/// Computes the broadcasted shape resulting from two input shapes according to standard NumPy/PyTorch rules.
pub fn broadcast_shapes(shape_a: &[usize], shape_b: &[usize]) -> Result<Vec<usize>> {
    let ndim_a = shape_a.len();
    let ndim_b = shape_b.len();
    let max_ndim = ndim_a.max(ndim_b);
    let mut result_shape = vec![0; max_ndim];

    for i in 0..max_ndim {
        let dim_a = if i < ndim_a {
            shape_a[ndim_a - 1 - i]
        } else {
            1
        };
        let dim_b = if i < ndim_b {
            shape_b[ndim_b - 1 - i]
        } else {
            1
        };

        if dim_a == dim_b {
            result_shape[max_ndim - 1 - i] = dim_a;
        } else if dim_a == 1 {
            result_shape[max_ndim - 1 - i] = dim_b;
        } else if dim_b == 1 {
            result_shape[max_ndim - 1 - i] = dim_a;
        } else {
            return Err(EngineError::BroadcastError {
                from: shape_a.to_vec(),
                to: shape_b.to_vec(),
            });
        }
    }

    Ok(result_shape)
}

/// Computes the broadcasted strides for expanding an existing shape & strides to a target shape.
/// Dimensions that are expanded from 1 have their stride set to 0.
pub fn broadcast_strides(
    current_shape: &[usize],
    current_strides: &[usize],
    target_shape: &[usize],
) -> Result<Vec<usize>> {
    let curr_ndim = current_shape.len();
    let target_ndim = target_shape.len();

    if curr_ndim > target_ndim {
        return Err(EngineError::BroadcastError {
            from: current_shape.to_vec(),
            to: target_shape.to_vec(),
        });
    }

    let mut new_strides = vec![0; target_ndim];
    let offset = target_ndim - curr_ndim;

    for i in 0..target_ndim {
        if i < offset {
            new_strides[i] = 0;
        } else {
            let curr_idx = i - offset;
            let curr_dim = current_shape[curr_idx];
            let target_dim = target_shape[i];

            if curr_dim == target_dim {
                new_strides[i] = current_strides[curr_idx];
            } else if curr_dim == 1 {
                new_strides[i] = 0;
            } else {
                return Err(EngineError::BroadcastError {
                    from: current_shape.to_vec(),
                    to: target_shape.to_vec(),
                });
            }
        }
    }

    Ok(new_strides)
}

/// Translates a flat index in [0, numel(shape)) to multi-dimensional coordinate indices.
#[inline(always)]
pub fn flat_to_multi_index(flat: usize, shape: &[usize], multi: &mut [usize]) {
    let mut rem = flat;
    for i in (0..shape.len()).rev() {
        let dim = shape[i];
        if dim == 0 {
            multi[i] = 0;
        } else {
            multi[i] = rem % dim;
            rem /= dim;
        }
    }
}

/// Translates multi-dimensional coordinate indices into linear offset in memory given strides and base offset.
#[inline(always)]
pub fn multi_index_to_offset(multi: &[usize], strides: &[usize], base_offset: usize) -> usize {
    let mut offset = base_offset;
    for (&idx, &stride) in multi.iter().zip(strides.iter()) {
        offset += idx * stride;
    }
    offset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strides_and_contiguity() {
        let shape = vec![2, 3, 4];
        let strides = compute_c_contiguous_strides(&shape);
        assert_eq!(strides, vec![12, 4, 1]);
        assert!(is_contiguous(&shape, &strides));
        assert_eq!(numel(&shape), 24);
    }

    #[test]
    fn test_broadcasting() {
        let a = vec![2, 1, 4];
        let b = vec![3, 4];
        let out = broadcast_shapes(&a, &b).unwrap();
        assert_eq!(out, vec![2, 3, 4]);

        let a_strides = compute_c_contiguous_strides(&a);
        let b_strides = broadcast_strides(&a, &a_strides, &out).unwrap();
        assert_eq!(b_strides, vec![4, 0, 1]);
    }
}
