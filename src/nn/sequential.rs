//! Sequential container for cascading layers.

use crate::autograd::Tensor;
use crate::error::Result;
use crate::nn::module::Module;

/// Sequential container chaining multiple neural network modules.
pub struct Sequential {
    pub layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    /// Creates an empty Sequential container.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Adds a module to the chain.
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: Module + 'static>(mut self, module: M) -> Self {
        self.layers.push(Box::new(module));
        self
    }

    /// Appends a module in-place.
    pub fn push<M: Module + 'static>(&mut self, module: M) {
        self.layers.push(Box::new(module));
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut current = input.clone();
        for layer in &self.layers {
            current = layer.forward(&current)?;
        }
        Ok(current)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }

    fn train(&mut self) {
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        for layer in &mut self.layers {
            layer.eval();
        }
    }
}
