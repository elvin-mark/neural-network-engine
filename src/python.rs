#![allow(clippy::useless_conversion)]

use crate::autograd::Tensor as RustTensor;
use crate::nn::activations::{ReLU as RustReLU, SiLU as RustSiLU, GELU as RustGELU};
use crate::nn::linear::Linear as RustLinear;
use crate::nn::module::Module;
use crate::nn::norm::{LayerNorm as RustLayerNorm, RMSNorm as RustRMSNorm};
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

#[pyclass(name = "LayerNorm")]
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

#[pyclass(name = "ReLU")]
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
    m.add_class::<PyTensor>()?;
    m.add_class::<PyLinear>()?;
    m.add_class::<PyLayerNorm>()?;
    m.add_class::<PyRMSNorm>()?;
    m.add_class::<PyReLU>()?;
    m.add_class::<PyGELU>()?;
    m.add_class::<PySiLU>()?;
    m.add_class::<PySGD>()?;
    m.add_class::<PyAdam>()?;
    m.add_class::<PyLossScaler>()?;

    Ok(())
}
