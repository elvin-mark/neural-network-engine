//! Pretty-printing for N-dimensional tensors.

use crate::tensor::RawTensor;
use std::fmt;

pub fn format_tensor(tensor: &RawTensor, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let shape = tensor.shape();
    if shape.is_empty() {
        return write!(f, "Tensor({}, shape=[])", tensor.item());
    }

    writeln!(f, "Tensor(")?;
    format_recursive(tensor, &[], 0, f)?;
    write!(f, ",\n  shape={:?})", shape)
}

fn format_recursive(
    tensor: &RawTensor,
    prefix_indices: &[usize],
    depth: usize,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    let shape = tensor.shape();
    let current_dim = shape[depth];
    let indent = " ".repeat((depth + 1) * 2);

    if depth == shape.len() - 1 {
        // Innermost 1D array
        write!(f, "{}[", indent)?;
        let limit = 8;
        if current_dim <= limit {
            for i in 0..current_dim {
                let mut idx = prefix_indices.to_vec();
                idx.push(i);
                let val = tensor.get(&idx);
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:8.4}", val)?;
            }
        } else {
            for i in 0..3 {
                let mut idx = prefix_indices.to_vec();
                idx.push(i);
                let val = tensor.get(&idx);
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:8.4}", val)?;
            }
            write!(f, ",   ...  ")?;
            for i in (current_dim - 3)..current_dim {
                let mut idx = prefix_indices.to_vec();
                idx.push(i);
                let val = tensor.get(&idx);
                write!(f, ", {:8.4}", val)?;
            }
        }
        write!(f, "]")?;
    } else {
        writeln!(f, "{}[", indent)?;
        let limit = 6;
        if current_dim <= limit {
            for i in 0..current_dim {
                let mut next_indices = prefix_indices.to_vec();
                next_indices.push(i);
                format_recursive(tensor, &next_indices, depth + 1, f)?;
                if i < current_dim - 1 {
                    writeln!(f, ",")?;
                }
            }
        } else {
            for i in 0..2 {
                let mut next_indices = prefix_indices.to_vec();
                next_indices.push(i);
                format_recursive(tensor, &next_indices, depth + 1, f)?;
                writeln!(f, ",")?;
            }
            writeln!(f, "{}  ...", indent)?;
            for i in (current_dim - 2)..current_dim {
                let mut next_indices = prefix_indices.to_vec();
                next_indices.push(i);
                format_recursive(tensor, &next_indices, depth + 1, f)?;
                if i < current_dim - 1 {
                    writeln!(f, ",")?;
                }
            }
        }
        write!(f, "\n{}]", indent)?;
    }
    Ok(())
}
