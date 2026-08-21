//! 2D Convolution neural network layer.

use crate::autograd::Tensor;
use crate::error::Result;
use crate::nn::module::Module;
use crate::tensor::conv::Conv2dParams;

/// 2D Convolution layer over an input signal composed of several input planes.
#[derive(Clone)]
pub struct Conv2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub params: Conv2dParams,
}

impl Conv2d {
    /// Creates a new Conv2d layer with default stride=1 and padding=0.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: (usize, usize)) -> Self {
        Self::with_options(
            in_channels,
            out_channels,
            kernel_size,
            (1, 1),
            (0, 0),
            (1, 1),
            true,
        )
    }

    /// Creates a Conv2d layer with custom stride, padding, dilation, and bias configuration.
    pub fn with_options(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        has_bias: bool,
    ) -> Self {
        let fan_in = in_channels * kernel_size.0 * kernel_size.1;
        let weight = Tensor::kaiming_uniform(
            &[out_channels, in_channels, kernel_size.0, kernel_size.1],
            fan_in,
            true,
        );

        let bias = if has_bias {
            let bound = 1.0 / (fan_in as f32).sqrt();
            Some(Tensor::uniform(&[out_channels], -bound, bound, true))
        } else {
            None
        };

        Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            params: Conv2dParams {
                stride,
                padding,
                dilation,
            },
        }
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.conv2d(&self.weight, self.bias.as_ref(), self.params)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }
}
