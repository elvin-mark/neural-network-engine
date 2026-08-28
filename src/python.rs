#![allow(clippy::useless_conversion)]

use crate::autograd::Tensor as RustTensor;
use crate::nn::activations::{
    LeakyReLU as RustLeakyReLU, ReLU as RustReLU, SiLU as RustSiLU, Sigmoid as RustSigmoid,
    Tanh as RustTanh, GELU as RustGELU,
};
use crate::nn::attention::MultiHeadAttention as RustMultiHeadAttention;
use crate::nn::conv::Conv2d as RustConv2d;
use crate::nn::dropout::Dropout as RustDropout;
use crate::nn::embedding::Embedding as RustEmbedding;
use crate::nn::flash_attention::FlashAttention as RustFlashAttention;
use crate::nn::linear::Linear as RustLinear;
use crate::nn::llama::SwiGLU as RustSwiGLU;
use crate::nn::loss::{CrossEntropyLoss as RustCrossEntropyLoss, MSELoss as RustMSELoss};
use crate::nn::module::Module;
use crate::nn::moe::{MoEConfig as RustMoEConfig, MoELayer as RustMoELayer};
use crate::nn::norm::{
    BatchNorm1d as RustBatchNorm1d, BatchNorm2d as RustBatchNorm2d, LayerNorm as RustLayerNorm,
    RMSNorm as RustRMSNorm,
};
use crate::nn::pooling::MaxPool2d as RustMaxPool2d;
use crate::nn::resnet::{ResNet as RustResNet, ResidualBlock as RustResidualBlock};
use crate::nn::transformer::{
    TransformerBlock as RustTransformerBlock, TransformerLM as RustTransformerLM,
};
use crate::optim::adam::Adam as RustAdam;
use crate::optim::amp::LossScaler as RustLossScaler;
use crate::optim::sgd::SGD as RustSGD;
use crate::optim::Optimizer;
use crate::tensor::RawTensor;

use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// =============================================================================
// PyTensor Wrapper
// =============================================================================

/// Dynamic automatic differentiation Tensor with NumPy interoperability.
#[pyclass(name = "Tensor")]
#[derive(Clone)]
pub struct PyTensor {
    pub(crate) inner: RustTensor,
}

#[pymethods]
impl PyTensor {
    /// Creates a tensor from a NumPy ndarray.
    #[classmethod]
    #[pyo3(signature = (array, requires_grad=false))]
    fn from_numpy(
        _cls: &Bound<'_, pyo3::types::PyType>,
        array: PyReadonlyArrayDyn<f32>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        let shape: Vec<usize> = array.shape().to_vec();
        let slice = array
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("Non-contiguous array: {}", e)))?;
        let raw = RawTensor::from_slice(slice, &shape);
        Ok(PyTensor {
            inner: RustTensor::new(raw, requires_grad),
        })
    }

    /// Converts this tensor to a NumPy ndarray.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let raw = self.inner.data();
        let contig = raw.to_contiguous();
        let shape = contig.shape().to_vec();
        let slice = contig.as_slice().to_vec();

        let array = ArrayD::from_shape_vec(IxDyn(&shape), slice)
            .map_err(|e| PyValueError::new_err(format!("Shape error: {}", e)))?;
        Ok(array.into_pyarray_bound(py))
    }

    /// Creates a tensor filled with zeros.
    #[classmethod]
    #[pyo3(signature = (shape, requires_grad=false))]
    fn zeros(
        _cls: &Bound<'_, pyo3::types::PyType>,
        shape: Vec<usize>,
        requires_grad: bool,
    ) -> Self {
        PyTensor {
            inner: RustTensor::zeros(&shape, requires_grad),
        }
    }

    /// Creates a tensor filled with ones.
    #[classmethod]
    #[pyo3(signature = (shape, requires_grad=false))]
    fn ones(_cls: &Bound<'_, pyo3::types::PyType>, shape: Vec<usize>, requires_grad: bool) -> Self {
        PyTensor {
            inner: RustTensor::ones(&shape, requires_grad),
        }
    }

    /// Creates a tensor with random normal values.
    #[classmethod]
    #[pyo3(signature = (shape, mean=0.0, std=1.0, requires_grad=false))]
    fn randn(
        _cls: &Bound<'_, pyo3::types::PyType>,
        shape: Vec<usize>,
        mean: f32,
        std: f32,
        requires_grad: bool,
    ) -> Self {
        PyTensor {
            inner: RustTensor::randn(&shape, mean, std, requires_grad),
        }
    }

    /// Returns the shape of the tensor as a Python list.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape()
    }

    /// Returns whether this tensor tracks gradients.
    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    /// Returns the gradient tensor if one has been computed.
    #[getter]
    fn grad(&self) -> Option<PyTensor> {
        self.inner.grad().map(|raw| PyTensor {
            inner: RustTensor::new(raw, false),
        })
    }

    /// Runs backward automatic differentiation from this scalar tensor.
    fn backward(&self) {
        self.inner.backward();
    }

    /// Returns the scalar item value if tensor has 1 element.
    fn item(&self) -> f32 {
        self.inner.item()
    }

    // Arithmetic operators
    fn __add__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.inner
            .add(&other.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __sub__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.inner
            .sub(&other.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __mul__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.inner
            .mul(&other.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __truediv__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.inner
            .div(&other.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __matmul__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.inner
            .matmul(&other.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __neg__(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.neg(),
        }
    }

    // Activations and Reductions
    fn relu(&self) -> PyResult<PyTensor> {
        self.inner
            .relu()
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn gelu(&self) -> PyResult<PyTensor> {
        self.inner
            .gelu()
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn silu(&self) -> PyResult<PyTensor> {
        self.inner
            .silu()
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn sigmoid(&self) -> PyResult<PyTensor> {
        self.inner
            .sigmoid()
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn tanh(&self) -> PyResult<PyTensor> {
        self.inner
            .tanh()
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (axis=None))]
    fn softmax(&self, axis: Option<usize>) -> PyResult<PyTensor> {
        let ax = axis.unwrap_or_else(|| self.inner.shape().len().saturating_sub(1));
        self.inner
            .softmax(ax)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn sum(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.sum_all(),
        }
    }

    fn mean(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.mean_all(),
        }
    }

    fn reshape(&self, shape: Vec<usize>) -> PyResult<PyTensor> {
        self.inner
            .reshape(&shape)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn transpose(&self, dim0: usize, dim1: usize) -> PyResult<PyTensor> {
        self.inner
            .transpose(dim0, dim1)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, requires_grad={})",
            self.shape(),
            self.requires_grad()
        )
    }
}

// =============================================================================
// Neural Network Layers
// =============================================================================

#[pyclass(name = "Linear")]
#[derive(Clone)]
pub struct PyLinear {
    pub(crate) inner: RustLinear,
}

#[pymethods]
impl PyLinear {
    #[new]
    #[pyo3(signature = (in_features, out_features, bias=true))]
    fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        PyLinear {
            inner: RustLinear::with_bias(in_features, out_features, bias),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.inner
            .forward(&x.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

#[pyclass(name = "Conv2d")]
#[derive(Clone)]
pub struct PyConv2d {
    pub(crate) inner: RustConv2d,
}

#[pymethods]
impl PyConv2d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1), bias=true))]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        bias: bool,
    ) -> Self {
        PyConv2d {
            inner: RustConv2d::with_options(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                bias,
            ),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.inner
            .forward(&x.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

#[pyclass(name = "MaxPool2d")]
#[derive(Clone)]
pub struct PyMaxPool2d {
    pub(crate) inner: RustMaxPool2d,
}

#[pymethods]
impl PyMaxPool2d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=None))]
    fn new(kernel_size: (usize, usize), stride: Option<(usize, usize)>) -> Self {
        let strd = stride.unwrap_or(kernel_size);
        PyMaxPool2d {
            inner: RustMaxPool2d::new(kernel_size, strd),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.inner
            .forward(&x.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
}

#[pyclass(name = "Embedding")]
#[derive(Clone)]
pub struct PyEmbedding {
    pub(crate) inner: RustEmbedding,
}

#[pymethods]
impl PyEmbedding {
    #[new]
    fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        PyEmbedding {
            inner: RustEmbedding::new(num_embeddings, embedding_dim),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.inner
            .forward(&x.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

#[pyclass(name = "Dropout")]
#[derive(Clone)]
pub struct PyDropout {
    pub(crate) inner: RustDropout,
}

#[pymethods]
impl PyDropout {
    #[new]
    #[pyo3(signature = (p=0.5))]
    fn new(p: f32) -> Self {
        PyDropout {
            inner: RustDropout::new(p),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.inner
            .forward(&x.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    #[setter]
    fn set_training(&mut self, is_training: bool) {
        self.inner.is_training = is_training;
    }
}

#[pyclass(name = "LayerNorm")]
#[derive(Clone)]
pub struct PyLayerNorm {
    pub(crate) inner: RustLayerNorm,
}

#[pymethods]
impl PyLayerNorm {
    #[new]
    #[pyo3(signature = (normalized_shape, eps=1e-5))]
    fn new(normalized_shape: usize, eps: f32) -> Self {
        PyLayerNorm {
            inner: RustLayerNorm::with_eps(normalized_shape, eps),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.inner
            .forward(&x.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

#[pyclass(name = "RMSNorm")]
#[derive(Clone)]
pub struct PyRMSNorm {
    pub(crate) inner: RustRMSNorm,
}

#[pymethods]
impl PyRMSNorm {
    #[new]
    #[pyo3(signature = (dim, eps=1e-6))]
    fn new(dim: usize, eps: f32) -> Self {
        PyRMSNorm {
            inner: RustRMSNorm::with_eps(dim, eps),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.inner
            .forward(&x.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

#[pyclass(name = "BatchNorm1d")]
#[derive(Clone)]
pub struct PyBatchNorm1d {
    pub(crate) inner: RustBatchNorm1d,
}

#[pymethods]
impl PyBatchNorm1d {
    #[new]
    fn new(num_features: usize) -> Self {
        PyBatchNorm1d {
            inner: RustBatchNorm1d::new(num_features),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.inner
            .forward(&x.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

#[pyclass(name = "BatchNorm2d")]
#[derive(Clone)]
pub struct PyBatchNorm2d {
    pub(crate) inner: RustBatchNorm2d,
}

#[pymethods]
impl PyBatchNorm2d {
    #[new]
    fn new(num_features: usize) -> Self {
        PyBatchNorm2d {
            inner: RustBatchNorm2d::new(num_features),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.inner
            .forward(&x.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

#[pyclass(name = "MultiHeadAttention")]
#[derive(Clone)]
pub struct PyMultiHeadAttention {
    pub(crate) inner: RustMultiHeadAttention,
}

#[pymethods]
impl PyMultiHeadAttention {
    #[new]
    #[pyo3(signature = (d_model, num_heads, is_causal=true))]
    fn new(d_model: usize, num_heads: usize, is_causal: bool) -> Self {
        PyMultiHeadAttention {
            inner: RustMultiHeadAttention::new(d_model, num_heads, is_causal),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.inner
            .forward(&x.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

#[pyclass(name = "FlashAttention")]
#[derive(Clone)]
pub struct PyFlashAttention {
    pub(crate) inner: RustFlashAttention,
}

#[pymethods]
impl PyFlashAttention {
    #[new]
    #[pyo3(signature = (d_model, num_heads, is_causal=true))]
    fn new(d_model: usize, num_heads: usize, is_causal: bool) -> Self {
        PyFlashAttention {
            inner: RustFlashAttention::new(d_model, num_heads, is_causal),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.inner
            .forward(&x.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

#[pyclass(name = "SwiGLU")]
#[derive(Clone)]
pub struct PySwiGLU {
    pub(crate) inner: RustSwiGLU,
}

#[pymethods]
impl PySwiGLU {
    #[new]
    fn new(d_model: usize, hidden_dim: usize) -> Self {
        PySwiGLU {
            inner: RustSwiGLU::new(d_model, hidden_dim),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.inner
            .forward(&x.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

#[pyclass(name = "MoELayer")]
#[derive(Clone)]
pub struct PyMoELayer {
    pub(crate) inner: RustMoELayer,
}

#[pymethods]
impl PyMoELayer {
    #[new]
    #[pyo3(signature = (d_model, hidden_dim, num_experts=8, top_k=2, aux_loss_coeff=0.01))]
    fn new(
        d_model: usize,
        hidden_dim: usize,
        num_experts: usize,
        top_k: usize,
        aux_loss_coeff: f32,
    ) -> Self {
        let config = RustMoEConfig {
            d_model,
            hidden_dim,
            num_experts,
            top_k,
            aux_loss_coeff,
        };
        PyMoELayer {
            inner: RustMoELayer::new(config),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.inner
            .forward(&x.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn forward_with_aux(&self, x: &PyTensor) -> PyResult<(PyTensor, PyTensor)> {
        self.inner
            .forward_with_aux(&x.inner)
            .map(|(out, aux)| (PyTensor { inner: out }, PyTensor { inner: aux }))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

#[pyclass(name = "TransformerBlock")]
pub struct PyTransformerBlock {
    pub(crate) inner: RustTransformerBlock,
}

#[pymethods]
impl PyTransformerBlock {
    #[new]
    #[pyo3(signature = (d_model, num_heads, is_causal=true))]
    fn new(d_model: usize, num_heads: usize, is_causal: bool) -> Self {
        PyTransformerBlock {
            inner: RustTransformerBlock::new(d_model, num_heads, is_causal),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.inner
            .forward(&x.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

#[pyclass(name = "TransformerLM")]
pub struct PyTransformerLM {
    pub(crate) inner: RustTransformerLM,
}

#[pymethods]
impl PyTransformerLM {
    #[new]
    #[pyo3(signature = (vocab_size, max_seq_len, d_model, num_heads, num_layers))]
    fn new(
        vocab_size: usize,
        max_seq_len: usize,
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
    ) -> Self {
        PyTransformerLM {
            inner: RustTransformerLM::new(vocab_size, max_seq_len, d_model, num_heads, num_layers),
        }
    }

    fn forward(&self, token_indices: &PyTensor) -> PyResult<PyTensor> {
        let contig = token_indices.inner.data().to_contiguous();
        let shape = contig.shape();
        let (b, t) = if shape.len() == 2 {
            (shape[0], shape[1])
        } else if shape.len() == 1 {
            (1, shape[0])
        } else {
            return Err(PyValueError::new_err(format!(
                "Expected 1D or 2D token indices, got shape {:?}",
                shape
            )));
        };

        let slice = contig.as_slice();
        let indices: Vec<usize> = slice.iter().map(|&x| x as usize).collect();

        self.inner
            .forward_tokens(&indices, b, t)
            .map(|out| PyTensor { inner: out })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __call__(&self, token_indices: &PyTensor) -> PyResult<PyTensor> {
        self.forward(token_indices)
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

#[pyclass(name = "ResidualBlock")]
pub struct PyResidualBlock {
    pub(crate) inner: RustResidualBlock,
}

#[pymethods]
impl PyResidualBlock {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, stride=1))]
    fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        PyResidualBlock {
            inner: RustResidualBlock::new(in_channels, out_channels, stride),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.inner
            .forward(&x.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

#[pyclass(name = "ResNet18")]
pub struct PyResNet18 {
    pub(crate) inner: RustResNet,
}

#[pymethods]
impl PyResNet18 {
    #[new]
    #[pyo3(signature = (num_classes=10, in_channels=3, cifar_stem=true))]
    fn new(num_classes: usize, in_channels: usize, cifar_stem: bool) -> Self {
        PyResNet18 {
            inner: if cifar_stem {
                RustResNet::cifar_resnet18(in_channels, num_classes)
            } else {
                RustResNet::resnet18(in_channels, num_classes)
            },
        }
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.inner
            .forward(&x.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

// =============================================================================
// Activations & Loss Functions
// =============================================================================

#[pyclass(name = "ReLU")]
#[derive(Clone)]
pub struct PyReLU;

#[pymethods]
impl PyReLU {
    #[new]
    fn new() -> Self {
        PyReLU
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        RustReLU
            .forward(&x.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
}

#[pyclass(name = "GELU")]
#[derive(Clone)]
pub struct PyGELU;

#[pymethods]
impl PyGELU {
    #[new]
    fn new() -> Self {
        PyGELU
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        RustGELU
            .forward(&x.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
}

#[pyclass(name = "SiLU")]
#[derive(Clone)]
pub struct PySiLU;

#[pymethods]
impl PySiLU {
    #[new]
    fn new() -> Self {
        PySiLU
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        RustSiLU
            .forward(&x.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
}

#[pyclass(name = "Sigmoid")]
#[derive(Clone)]
pub struct PySigmoid;

#[pymethods]
impl PySigmoid {
    #[new]
    fn new() -> Self {
        PySigmoid
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        RustSigmoid
            .forward(&x.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
}

#[pyclass(name = "Tanh")]
#[derive(Clone)]
pub struct PyTanh;

#[pymethods]
impl PyTanh {
    #[new]
    fn new() -> Self {
        PyTanh
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        RustTanh
            .forward(&x.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
}

#[pyclass(name = "LeakyReLU")]
#[derive(Clone)]
pub struct PyLeakyReLU {
    pub(crate) inner: RustLeakyReLU,
}

#[pymethods]
impl PyLeakyReLU {
    #[new]
    #[pyo3(signature = (negative_slope=0.01))]
    fn new(negative_slope: f32) -> Self {
        PyLeakyReLU {
            inner: RustLeakyReLU::new(negative_slope),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.inner
            .forward(&x.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
}

#[pyclass(name = "MSELoss")]
pub struct PyMSELoss;

#[pymethods]
impl PyMSELoss {
    #[new]
    fn new() -> Self {
        PyMSELoss
    }

    fn forward(&self, pred: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        RustMSELoss::forward(&pred.inner, &target.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __call__(&self, pred: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        self.forward(pred, target)
    }
}

#[pyclass(name = "CrossEntropyLoss")]
pub struct PyCrossEntropyLoss;

#[pymethods]
impl PyCrossEntropyLoss {
    #[new]
    fn new() -> Self {
        PyCrossEntropyLoss
    }

    fn forward(&self, logits: &PyTensor, targets: &PyTensor) -> PyResult<PyTensor> {
        let targets_raw = targets.inner.data();
        let targets_contig = targets_raw.to_contiguous();
        let slice = targets_contig.as_slice();
        let mut indices = Vec::with_capacity(slice.len());
        for &val in slice {
            indices.push(val as usize);
        }

        RustCrossEntropyLoss::forward_with_indices(&logits.inner, &indices)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __call__(&self, logits: &PyTensor, targets: &PyTensor) -> PyResult<PyTensor> {
        self.forward(logits, targets)
    }
}

// =============================================================================
// Optimizers
// =============================================================================

#[pyclass(name = "SGD")]
pub struct PySGD {
    pub(crate) inner: RustSGD,
}

#[pymethods]
impl PySGD {
    #[new]
    #[pyo3(signature = (params, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=false))]
    fn new(
        params: Vec<PyTensor>,
        lr: f32,
        momentum: f32,
        weight_decay: f32,
        nesterov: bool,
    ) -> Self {
        let rust_params: Vec<RustTensor> = params.into_iter().map(|p| p.inner).collect();
        let sgd = RustSGD::new(rust_params, lr)
            .with_momentum(momentum)
            .with_weight_decay(weight_decay)
            .with_nesterov(nesterov);
        PySGD { inner: sgd }
    }

    fn step(&mut self) -> PyResult<()> {
        self.inner
            .step()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn zero_grad(&self) {
        self.inner.zero_grad();
    }

    #[getter]
    fn lr(&self) -> f32 {
        self.inner.get_lr()
    }

    #[setter]
    fn set_lr(&mut self, lr: f32) {
        self.inner.set_lr(lr);
    }
}

#[pyclass(name = "Adam")]
pub struct PyAdam {
    pub(crate) inner: RustAdam,
}

#[pymethods]
impl PyAdam {
    #[new]
    #[pyo3(signature = (params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0))]
    fn new(
        params: Vec<PyTensor>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> Self {
        let rust_params: Vec<RustTensor> = params.into_iter().map(|p| p.inner).collect();
        let adam = RustAdam::new(rust_params, lr)
            .with_betas(beta1, beta2)
            .with_eps(eps)
            .with_weight_decay(weight_decay);
        PyAdam { inner: adam }
    }

    fn step(&mut self) -> PyResult<()> {
        self.inner
            .step()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn zero_grad(&self) {
        self.inner.zero_grad();
    }

    #[getter]
    fn lr(&self) -> f32 {
        self.inner.get_lr()
    }

    #[setter]
    fn set_lr(&mut self, lr: f32) {
        self.inner.set_lr(lr);
    }
}

#[pyclass(name = "LossScaler")]
pub struct PyLossScaler {
    pub(crate) inner: RustLossScaler,
}

#[pymethods]
impl PyLossScaler {
    #[new]
    #[pyo3(signature = (init_scale=1024.0))]
    fn new(init_scale: f32) -> Self {
        PyLossScaler {
            inner: RustLossScaler::new(init_scale),
        }
    }

    fn current_scale(&self) -> f32 {
        self.inner.current_scale()
    }

    fn scale(&self, loss: &PyTensor) -> PyResult<PyTensor> {
        self.inner
            .scale(&loss.inner)
            .map(|t| PyTensor { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn step_adam(&self, opt: &mut PyAdam) -> PyResult<bool> {
        self.inner
            .step(&mut opt.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn step_sgd(&self, opt: &mut PySGD) -> PyResult<bool> {
        self.inner
            .step(&mut opt.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

// =============================================================================
// Python Module Declaration
// =============================================================================

#[pymodule]
fn neural_network_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core Tensor
    m.add_class::<PyTensor>()?;

    // NN Layers
    m.add_class::<PyLinear>()?;
    m.add_class::<PyConv2d>()?;
    m.add_class::<PyMaxPool2d>()?;
    m.add_class::<PyEmbedding>()?;
    m.add_class::<PyDropout>()?;
    m.add_class::<PyLayerNorm>()?;
    m.add_class::<PyRMSNorm>()?;
    m.add_class::<PyBatchNorm1d>()?;
    m.add_class::<PyBatchNorm2d>()?;
    m.add_class::<PyMultiHeadAttention>()?;
    m.add_class::<PyFlashAttention>()?;
    m.add_class::<PySwiGLU>()?;
    m.add_class::<PyMoELayer>()?;
    m.add_class::<PyTransformerBlock>()?;
    m.add_class::<PyTransformerLM>()?;
    m.add_class::<PyResidualBlock>()?;
    m.add_class::<PyResNet18>()?;

    // Activations
    m.add_class::<PyReLU>()?;
    m.add_class::<PyGELU>()?;
    m.add_class::<PySiLU>()?;
    m.add_class::<PySigmoid>()?;
    m.add_class::<PyTanh>()?;
    m.add_class::<PyLeakyReLU>()?;

    // Losses
    m.add_class::<PyMSELoss>()?;
    m.add_class::<PyCrossEntropyLoss>()?;

    // Optimizers & AMP
    m.add_class::<PySGD>()?;
    m.add_class::<PyAdam>()?;
    m.add_class::<PyLossScaler>()?;

    Ok(())
}
