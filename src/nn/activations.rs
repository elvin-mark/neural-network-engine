//! Neural network activation layer modules.

use crate::autograd::Tensor;
use crate::error::Result;
use crate::nn::module::Module;

/// Rectified Linear Unit activation layer.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReLU;

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.relu()
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// Gaussian Error Linear Unit activation layer.
#[derive(Clone, Copy, Debug, Default)]
pub struct GELU;

impl Module for GELU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.gelu()
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// Sigmoid activation layer.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sigmoid;

impl Module for Sigmoid {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.sigmoid()
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// Hyperbolic Tangent activation layer.
#[derive(Clone, Copy, Debug, Default)]
pub struct Tanh;

impl Module for Tanh {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.tanh()
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// Leaky ReLU activation layer.
#[derive(Clone, Copy, Debug)]
pub struct LeakyReLU {
    pub negative_slope: f32,
}

impl LeakyReLU {
    pub fn new(negative_slope: f32) -> Self {
        Self { negative_slope }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self {
            negative_slope: 0.01,
        }
    }
}

impl Module for LeakyReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.leaky_relu(self.negative_slope)
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// Softmax activation layer.
#[derive(Clone, Copy, Debug)]
pub struct Softmax {
    pub axis: usize,
}

impl Softmax {
    pub fn new(axis: usize) -> Self {
        Self { axis }
    }
}

impl Module for Softmax {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.softmax(self.axis)
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}
