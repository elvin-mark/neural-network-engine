//! Dropout regularization layer.

use crate::autograd::Tensor;
use crate::error::Result;
use crate::nn::module::Module;
use crate::tensor::RawTensor;
use rand::Rng;

/// During training, randomly zeroes some of the elements of the input tensor with probability p.
#[derive(Clone, Debug)]
pub struct Dropout {
    pub p: f32,
    pub is_training: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout probability must be in [0.0, 1.0), got {}",
            p
        );
        Self {
            p,
            is_training: true,
        }
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.is_training || self.p == 0.0 {
            return Ok(input.clone());
        }

        let scale = 1.0 / (1.0 - self.p);
        let numel = input.numel();
        let mut rng = rand::thread_rng();

        let mask_data: Vec<f32> = (0..numel)
            .map(|_| {
                if rng.gen::<f32>() >= self.p {
                    scale
                } else {
                    0.0
                }
            })
            .collect();

        let mask_tensor = Tensor::new(RawTensor::from_vec(mask_data, input.shape()), false);
        input.mul(&mask_tensor)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }

    fn train(&mut self) {
        self.is_training = true;
    }

    fn eval(&mut self) {
        self.is_training = false;
    }
}
