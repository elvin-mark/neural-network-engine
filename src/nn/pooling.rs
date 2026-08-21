//! Spatial pooling layers (MaxPool2d and AvgPool2d).

use crate::autograd::Tensor;
use crate::error::Result;
use crate::nn::module::Module;

/// Applies a 2D max pooling over an input signal composed of several input planes.
#[derive(Clone, Debug)]
pub struct MaxPool2d {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
}

impl MaxPool2d {
    pub fn new(kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
        Self {
            kernel_size,
            stride,
        }
    }

    /// Creates a square MaxPool2d with matching kernel size and stride.
    pub fn square(size: usize) -> Self {
        Self {
            kernel_size: (size, size),
            stride: (size, size),
        }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.max_pool2d(self.kernel_size, self.stride)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}
