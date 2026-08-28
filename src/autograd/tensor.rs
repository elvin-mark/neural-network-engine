//! High-level autograd Tensor wrapper with operator overloads and automatic backward differentiation.

use crate::autograd::context::is_grad_enabled;
use crate::autograd::node::{unbroadcast_to, BackwardFn, TensorInner};
use crate::error::Result;
use crate::tensor::conv::{
    conv2d_backward, conv2d_forward, max_pool2d_backward, max_pool2d_forward, Conv2dParams,
};
use crate::tensor::matmul::matmul;
use crate::tensor::shape::{
    compute_c_contiguous_strides, flat_to_multi_index, multi_index_to_offset, numel,
};
use crate::tensor::RawTensor;
use std::collections::HashSet;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Dynamic automatic differentiation Tensor with reference-counted computational graph.
#[derive(Clone)]
pub struct Tensor {
    pub(crate) inner: Arc<TensorInner>,
}

impl Tensor {
    /// Creates a new leaf tensor.
    pub fn new(data: RawTensor, requires_grad: bool) -> Self {
        Self {
            inner: TensorInner::new(data, requires_grad),
        }
    }

    /// Creates a tensor initialized to zeros.
    pub fn zeros(shape: &[usize], requires_grad: bool) -> Self {
        Self::new(RawTensor::zeros(shape), requires_grad)
    }

    /// Creates a tensor initialized to ones.
    pub fn ones(shape: &[usize], requires_grad: bool) -> Self {
        Self::new(RawTensor::ones(shape), requires_grad)
    }

    /// Creates a scalar tensor.
    pub fn scalar(val: f32, requires_grad: bool) -> Self {
        Self::new(RawTensor::scalar(val), requires_grad)
    }

    /// Creates a tensor with random normal values.
    pub fn randn(shape: &[usize], mean: f32, std: f32, requires_grad: bool) -> Self {
        Self::new(RawTensor::randn(shape, mean, std), requires_grad)
    }

    /// Creates a tensor with random uniform values.
    pub fn uniform(shape: &[usize], low: f32, high: f32, requires_grad: bool) -> Self {
        Self::new(RawTensor::uniform(shape, low, high), requires_grad)
    }

    /// Kaiming uniform initialization.
    pub fn kaiming_uniform(shape: &[usize], fan_in: usize, requires_grad: bool) -> Self {
        Self::new(RawTensor::kaiming_uniform(shape, fan_in), requires_grad)
    }

    /// Kaiming normal initialization.
    pub fn kaiming_normal(shape: &[usize], fan_in: usize, requires_grad: bool) -> Self {
        Self::new(RawTensor::kaiming_normal(shape, fan_in), requires_grad)
    }

    /// Xavier uniform initialization.
    pub fn xavier_uniform(
        shape: &[usize],
        fan_in: usize,
        fan_out: usize,
        requires_grad: bool,
    ) -> Self {
        Self::new(
            RawTensor::xavier_uniform(shape, fan_in, fan_out),
            requires_grad,
        )
    }

    /// Xavier normal initialization.
    pub fn xavier_normal(
        shape: &[usize],
        fan_in: usize,
        fan_out: usize,
        requires_grad: bool,
    ) -> Self {
        Self::new(
            RawTensor::xavier_normal(shape, fan_in, fan_out),
            requires_grad,
        )
    }

    /// Creates a tensor from raw float vector and shape.
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Self {
        Self::new(RawTensor::from_vec(data, shape), requires_grad)
    }

    /// Creates a tensor from a float slice and shape.
    pub fn from_slice(data: &[f32], shape: &[usize], requires_grad: bool) -> Self {
        Self::new(RawTensor::from_slice(data, shape), requires_grad)
    }

    // --- State and Accessors ---

    /// Gets a cloned copy of the raw tensor data.
    pub fn data(&self) -> RawTensor {
        self.inner.data.read().unwrap().clone()
    }

    /// Gets a cloned copy of the accumulated gradient, if any.
    pub fn grad(&self) -> Option<RawTensor> {
        self.inner.grad.read().unwrap().clone()
    }

    /// Resets the accumulated gradient to None.
    pub fn zero_grad(&self) {
        *self.inner.grad.write().unwrap() = None;
    }

    /// Overwrites the gradient tensor.
    pub fn set_grad(&self, grad: Option<RawTensor>) {
        *self.inner.grad.write().unwrap() = grad;
    }

    /// Overwrites the underlying tensor data.
    pub fn set_data(&self, data: RawTensor) {
        *self.inner.data.write().unwrap() = data;
    }

    pub fn requires_grad(&self) -> bool {
        self.inner.requires_grad.load(Ordering::Relaxed)
    }

    pub fn set_requires_grad(&self, requires_grad: bool) {
        self.inner
            .requires_grad
            .store(requires_grad, Ordering::Relaxed);
    }

    pub fn shape(&self) -> Vec<usize> {
        self.inner.data.read().unwrap().shape().to_vec()
    }

    pub fn ndim(&self) -> usize {
        self.inner.data.read().unwrap().ndim()
    }

    pub fn numel(&self) -> usize {
        self.inner.data.read().unwrap().numel()
    }

    pub fn item(&self) -> f32 {
        self.inner.data.read().unwrap().item()
    }

    // --- Reverse-Mode Automatic Differentiation ---

    /// Runs reverse-mode automatic differentiation from this scalar tensor through the DAG.
    pub fn backward(&self) {
        let mut topo: Vec<Arc<TensorInner>> = Vec::new();
        let mut visited: HashSet<usize> = HashSet::new();

        fn build_topo(
            node: &Arc<TensorInner>,
            visited: &mut HashSet<usize>,
            topo: &mut Vec<Arc<TensorInner>>,
        ) {
            if !visited.contains(&node.id) {
                visited.insert(node.id);
                for parent in &node.parents {
                    build_topo(parent, visited, topo);
                }
                topo.push(node.clone());
            }
        }

        build_topo(&self.inner, &mut visited, &mut topo);

        // Ephemeral per-backward gradient map (Node ID -> Accumulated Gradient)
        let mut grads: std::collections::HashMap<usize, RawTensor> =
            std::collections::HashMap::new();
        grads.insert(self.inner.id, RawTensor::ones(&self.shape()));

        // Propagate backwards through topologically sorted nodes
        for node in topo.iter().rev() {
            if let Some(current_grad) = grads.remove(&node.id) {
                // If this is a leaf node, accumulate into its persistent grad storage
                if node.parents.is_empty() && node.requires_grad.load(Ordering::Relaxed) {
                    node.accumulate_grad(current_grad.clone());
                }

                // If this node has backward_fn, propagate gradients to parents
                if let Some(ref backward_fn) = node.backward_fn {
                    let parent_grads = backward_fn(&current_grad);
                    for (parent, p_grad_opt) in node.parents.iter().zip(parent_grads) {
                        if let Some(p_grad) = p_grad_opt {
                            if parent.requires_grad.load(Ordering::Relaxed) {
                                match grads.get_mut(&parent.id) {
                                    Some(existing) => {
                                        *existing = existing
                                            .add(&p_grad)
                                            .expect("Failed to accumulate gradient in backward");
                                    }
                                    None => {
                                        grads.insert(parent.id, p_grad);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // --- Autograd Operations ---

    /// Elementwise addition.
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        let a_data = self.data();
        let b_data = other.data();
        let out_data = a_data.add(&b_data)?;

        if !is_grad_enabled() || (!self.requires_grad() && !other.requires_grad()) {
            return Ok(Tensor::new(out_data, false));
        }

        let a_shape = a_data.shape().to_vec();
        let b_shape = b_data.shape().to_vec();
        let a_req = self.requires_grad();
        let b_req = other.requires_grad();

        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let mut grads = Vec::new();
            if a_req {
                grads.push(Some(unbroadcast_to(grad, &a_shape).unwrap()));
            } else {
                grads.push(None);
            }
            if b_req {
                grads.push(Some(unbroadcast_to(grad, &b_shape).unwrap()));
            } else {
                grads.push(None);
            }
            grads
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(
                out_data,
                vec![self.inner.clone(), other.inner.clone()],
                backward_fn,
            ),
        })
    }

    /// Elementwise subtraction.
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        let a_data = self.data();
        let b_data = other.data();
        let out_data = a_data.sub(&b_data)?;

        if !is_grad_enabled() || (!self.requires_grad() && !other.requires_grad()) {
            return Ok(Tensor::new(out_data, false));
        }

        let a_shape = a_data.shape().to_vec();
        let b_shape = b_data.shape().to_vec();
        let a_req = self.requires_grad();
        let b_req = other.requires_grad();

        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let mut grads = Vec::new();
            if a_req {
                grads.push(Some(unbroadcast_to(grad, &a_shape).unwrap()));
            } else {
                grads.push(None);
            }
            if b_req {
                let neg_grad = grad.neg();
                grads.push(Some(unbroadcast_to(&neg_grad, &b_shape).unwrap()));
            } else {
                grads.push(None);
            }
            grads
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(
                out_data,
                vec![self.inner.clone(), other.inner.clone()],
                backward_fn,
            ),
        })
    }

    /// Elementwise multiplication (Hadamard product).
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        let a_data = self.data();
        let b_data = other.data();
        let out_data = a_data.mul(&b_data)?;

        if !is_grad_enabled() || (!self.requires_grad() && !other.requires_grad()) {
            return Ok(Tensor::new(out_data, false));
        }

        let a_shape = a_data.shape().to_vec();
        let b_shape = b_data.shape().to_vec();
        let a_req = self.requires_grad();
        let b_req = other.requires_grad();

        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let mut grads = Vec::new();
            if a_req {
                let da = grad.mul(&b_data).unwrap();
                grads.push(Some(unbroadcast_to(&da, &a_shape).unwrap()));
            } else {
                grads.push(None);
            }
            if b_req {
                let db = grad.mul(&a_data).unwrap();
                grads.push(Some(unbroadcast_to(&db, &b_shape).unwrap()));
            } else {
                grads.push(None);
            }
            grads
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(
                out_data,
                vec![self.inner.clone(), other.inner.clone()],
                backward_fn,
            ),
        })
    }

    /// Elementwise division.
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        let a_data = self.data();
        let b_data = other.data();
        let out_data = a_data.div(&b_data)?;

        if !is_grad_enabled() || (!self.requires_grad() && !other.requires_grad()) {
            return Ok(Tensor::new(out_data, false));
        }

        let a_shape = a_data.shape().to_vec();
        let b_shape = b_data.shape().to_vec();
        let a_req = self.requires_grad();
        let b_req = other.requires_grad();

        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let mut grads = Vec::new();
            if a_req {
                let da = grad.div(&b_data).unwrap();
                grads.push(Some(unbroadcast_to(&da, &a_shape).unwrap()));
            } else {
                grads.push(None);
            }
            if b_req {
                // db = -grad * a / (b^2)
                let b_sq = b_data.mul(&b_data).unwrap();
                let top = grad.mul(&a_data).unwrap().neg();
                let db = top.div(&b_sq).unwrap();
                grads.push(Some(unbroadcast_to(&db, &b_shape).unwrap()));
            } else {
                grads.push(None);
            }
            grads
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(
                out_data,
                vec![self.inner.clone(), other.inner.clone()],
                backward_fn,
            ),
        })
    }

    /// Adds a scalar value elementwise.
    pub fn add_scalar(&self, val: f32) -> Result<Tensor> {
        let scalar = Tensor::scalar(val, false);
        self.add(&scalar)
    }

    /// Multiplies by a scalar value elementwise.
    pub fn mul_scalar(&self, val: f32) -> Result<Tensor> {
        let scalar = Tensor::scalar(val, false);
        self.mul(&scalar)
    }

    /// Divides by a scalar value elementwise.
    pub fn div_scalar(&self, val: f32) -> Result<Tensor> {
        let scalar = Tensor::scalar(val, false);
        self.div(&scalar)
    }

    /// Unary negation.
    pub fn neg(&self) -> Tensor {
        let a_data = self.data();
        let out_data = a_data.neg();

        if !is_grad_enabled() || !self.requires_grad() {
            return Tensor::new(out_data, false);
        }

        let backward_fn: BackwardFn = Arc::new(|grad| vec![Some(grad.neg())]);

        Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        }
    }

    /// Matrix multiplication C = A * B.
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        let a_data = self.data();
        let b_data = other.data();
        let out_data = matmul(&a_data, &b_data)?;

        if !is_grad_enabled() || (!self.requires_grad() && !other.requires_grad()) {
            return Ok(Tensor::new(out_data, false));
        }

        let a_shape = a_data.shape().to_vec();
        let b_shape = b_data.shape().to_vec();
        let a_req = self.requires_grad();
        let b_req = other.requires_grad();

        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let mut grads = Vec::new();
            if a_req {
                // dA = grad * B^T
                let b_ndim = b_shape.len();
                let b_t = b_data.transpose(b_ndim - 2, b_ndim - 1).unwrap();
                let da = matmul(grad, &b_t).unwrap();
                grads.push(Some(unbroadcast_to(&da, &a_shape).unwrap()));
            } else {
                grads.push(None);
            }
            if b_req {
                // dB = A^T * grad
                let a_ndim = a_shape.len();
                let a_t = a_data.transpose(a_ndim - 2, a_ndim - 1).unwrap();
                let db = matmul(&a_t, grad).unwrap();
                grads.push(Some(unbroadcast_to(&db, &b_shape).unwrap()));
            } else {
                grads.push(None);
            }
            grads
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(
                out_data,
                vec![self.inner.clone(), other.inner.clone()],
                backward_fn,
            ),
        })
    }

    /// Multiplies self by transposed other (C = self * other^T) without allocating intermediate transposed buffers.
    pub fn matmul_transposed_b(&self, other: &Tensor) -> Result<Tensor> {
        let a_data = self.data();
        let b_data = other.data();
        let out_data = crate::tensor::matmul::matmul_transposed_b(&a_data, &b_data)?;

        if !is_grad_enabled() || (!self.requires_grad() && !other.requires_grad()) {
            return Ok(Tensor::new(out_data, false));
        }

        let a_shape = a_data.shape().to_vec();
        let b_shape = b_data.shape().to_vec();
        let a_req = self.requires_grad();
        let b_req = other.requires_grad();

        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let mut grads = Vec::new();
            if a_req {
                // dA = grad * B
                let da = matmul(grad, &b_data).unwrap();
                grads.push(Some(unbroadcast_to(&da, &a_shape).unwrap()));
            } else {
                grads.push(None);
            }
            if b_req {
                // dB = grad^T * A
                let g_ndim = grad.ndim();
                let g_t = grad.transpose(g_ndim - 2, g_ndim - 1).unwrap();
                let db = matmul(&g_t, &a_data).unwrap();
                grads.push(Some(unbroadcast_to(&db, &b_shape).unwrap()));
            } else {
                grads.push(None);
            }
            grads
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(
                out_data,
                vec![self.inner.clone(), other.inner.clone()],
                backward_fn,
            ),
        })
    }

    /// Sum reduction along an axis.
    pub fn sum(&self, axis: usize, keepdim: bool) -> Result<Tensor> {
        let a_data = self.data();
        let out_data = a_data.sum(axis, keepdim)?;

        if !is_grad_enabled() || !self.requires_grad() {
            return Ok(Tensor::new(out_data, false));
        }

        let in_shape = a_data.shape().to_vec();
        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let mut expanded_grad = grad.clone();
            if !keepdim {
                expanded_grad = expanded_grad.unsqueeze(axis).unwrap();
            }
            // Expand to input shape
            let zeros = RawTensor::zeros(&in_shape);
            let b_grad = zeros.add(&expanded_grad).unwrap();
            vec![Some(b_grad)]
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        })
    }

    /// Mean reduction along an axis.
    pub fn mean(&self, axis: usize, keepdim: bool) -> Result<Tensor> {
        let a_data = self.data();
        let axis_size = a_data.shape()[axis] as f32;
        let out_data = a_data.mean(axis, keepdim)?;

        if !is_grad_enabled() || !self.requires_grad() {
            return Ok(Tensor::new(out_data, false));
        }

        let in_shape = a_data.shape().to_vec();
        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let mut expanded_grad = grad.clone();
            if !keepdim {
                expanded_grad = expanded_grad.unsqueeze(axis).unwrap();
            }
            let scaled_grad = expanded_grad.div_scalar(axis_size).unwrap();
            let zeros = RawTensor::zeros(&in_shape);
            let b_grad = zeros.add(&scaled_grad).unwrap();
            vec![Some(b_grad)]
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        })
    }

    /// Sum of all elements in the tensor to a scalar.
    pub fn sum_all(&self) -> Tensor {
        let sum_val = self.data().sum_all();
        let out_data = RawTensor::scalar(sum_val);

        if !is_grad_enabled() || !self.requires_grad() {
            return Tensor::new(out_data, false);
        }

        let in_shape = self.shape();
        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let g = grad.item();
            let grad_tensor = RawTensor::full(&in_shape, g);
            vec![Some(grad_tensor)]
        });

        Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        }
    }

    /// Arithmetic mean of all elements in the tensor to a scalar.
    pub fn mean_all(&self) -> Tensor {
        let mean_val = self.data().mean_all();
        let out_data = RawTensor::scalar(mean_val);

        if !is_grad_enabled() || !self.requires_grad() {
            return Tensor::new(out_data, false);
        }

        let in_shape = self.shape();
        let count = in_shape.iter().product::<usize>() as f32;

        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let g = grad.item() / count;
            let grad_tensor = RawTensor::full(&in_shape, g);
            vec![Some(grad_tensor)]
        });

        Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        }
    }

    /// Natural exponential.
    pub fn exp(&self) -> Result<Tensor> {
        let a_data = self.data();
        let out_data = a_data.exp()?;

        if !is_grad_enabled() || !self.requires_grad() {
            return Ok(Tensor::new(out_data, false));
        }

        let saved_out = out_data.clone();
        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let d = grad.mul(&saved_out).unwrap();
            vec![Some(d)]
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        })
    }

    /// Natural logarithm.
    pub fn log(&self) -> Result<Tensor> {
        let a_data = self.data();
        let out_data = a_data.log()?;

        if !is_grad_enabled() || !self.requires_grad() {
            return Ok(Tensor::new(out_data, false));
        }

        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let d = grad.div(&a_data).unwrap();
            vec![Some(d)]
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        })
    }

    /// Square root elementwise.
    pub fn sqrt(&self) -> Result<Tensor> {
        let a_data = self.data();
        let out_data = a_data.sqrt()?;

        if !is_grad_enabled() || !self.requires_grad() {
            return Ok(Tensor::new(out_data, false));
        }

        let saved_out = out_data.clone();
        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let two_out = saved_out.mul_scalar(2.0).unwrap();
            let d = grad.div(&two_out).unwrap();
            vec![Some(d)]
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        })
    }

    /// Power function with scalar exponent.
    pub fn powf(&self, exp: f32) -> Result<Tensor> {
        let a_data = self.data();
        let out_data = a_data.powf(exp)?;

        if !is_grad_enabled() || !self.requires_grad() {
            return Ok(Tensor::new(out_data, false));
        }

        let backward_fn: BackwardFn = Arc::new(move |grad| {
            // d = grad * exp * a^(exp - 1)
            let a_pow = a_data.powf(exp - 1.0).unwrap();
            let d = grad.mul(&a_pow).unwrap().mul_scalar(exp).unwrap();
            vec![Some(d)]
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        })
    }

    /// Hyperbolic tangent activation.
    pub fn tanh(&self) -> Result<Tensor> {
        let a_data = self.data();
        let out_data = a_data.tanh()?;

        if !is_grad_enabled() || !self.requires_grad() {
            return Ok(Tensor::new(out_data, false));
        }

        let saved_out = out_data.clone();
        let backward_fn: BackwardFn = Arc::new(move |grad| {
            // d = grad * (1 - tanh^2)
            let t2 = saved_out.mul(&saved_out).unwrap();
            let ones = RawTensor::ones(saved_out.shape());
            let one_minus_t2 = ones.sub(&t2).unwrap();
            let d = grad.mul(&one_minus_t2).unwrap();
            vec![Some(d)]
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        })
    }

    /// Sigmoid activation.
    pub fn sigmoid(&self) -> Result<Tensor> {
        let a_data = self.data();
        let out_data = a_data.sigmoid()?;

        if !is_grad_enabled() || !self.requires_grad() {
            return Ok(Tensor::new(out_data, false));
        }

        let saved_out = out_data.clone();
        let backward_fn: BackwardFn = Arc::new(move |grad| {
            // d = grad * s * (1 - s)
            let ones = RawTensor::ones(saved_out.shape());
            let one_minus_s = ones.sub(&saved_out).unwrap();
            let s_prime = saved_out.mul(&one_minus_s).unwrap();
            let d = grad.mul(&s_prime).unwrap();
            vec![Some(d)]
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        })
    }

    /// Rectified Linear Unit (ReLU).
    pub fn relu(&self) -> Result<Tensor> {
        let a_data = self.data();
        let out_data = a_data.relu()?;

        if !is_grad_enabled() || !self.requires_grad() {
            return Ok(Tensor::new(out_data, false));
        }

        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let mask = a_data.unary_op(|x| if x > 0.0 { 1.0 } else { 0.0 });
            let d = grad.mul(&mask).unwrap();
            vec![Some(d)]
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        })
    }

    /// Gaussian Error Linear Unit (GELU).
    pub fn gelu(&self) -> Result<Tensor> {
        let a_data = self.data();
        let out_data = a_data.gelu()?;

        if !is_grad_enabled() || !self.requires_grad() {
            return Ok(Tensor::new(out_data, false));
        }

        const SQRT_2_OVER_PI: f32 = 0.797_884_6;
        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let d_gelu = a_data.unary_op(|x| {
                let x3 = x * x * x;
                let inner = SQRT_2_OVER_PI * (x + 0.044715 * x3);
                let tanh_val = inner.tanh();
                let sech2 = 1.0 - tanh_val * tanh_val;
                0.5 * (1.0 + tanh_val)
                    + 0.5 * x * sech2 * SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * x * x)
            });
            let d = grad.mul(&d_gelu).unwrap();
            vec![Some(d)]
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        })
    }

    /// Leaky ReLU activation.
    pub fn leaky_relu(&self, negative_slope: f32) -> Result<Tensor> {
        let a_data = self.data();
        let out_data = a_data.leaky_relu(negative_slope)?;

        if !is_grad_enabled() || !self.requires_grad() {
            return Ok(Tensor::new(out_data, false));
        }

        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let mask = a_data.unary_op(|x| if x > 0.0 { 1.0 } else { negative_slope });
            let d = grad.mul(&mask).unwrap();
            vec![Some(d)]
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        })
    }

    /// Sigmoid Linear Unit (SiLU / Swish): x * sigmoid(x).
    pub fn silu(&self) -> Result<Tensor> {
        let a_data = self.data();
        let out_data = a_data.silu()?;

        if !is_grad_enabled() || !self.requires_grad() {
            return Ok(Tensor::new(out_data, false));
        }

        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let d_silu = a_data.unary_op(|x| {
                let sig = 1.0 / (1.0 + (-x).exp());
                sig * (1.0 + x * (1.0 - sig))
            });
            let d = grad.mul(&d_silu).unwrap();
            vec![Some(d)]
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        })
    }

    /// Numerically stable Log-Softmax along an axis.
    pub fn log_softmax(&self, axis: usize) -> Result<Tensor> {
        let a_data = self.data();
        let out_data = a_data.log_softmax(axis)?;

        if !is_grad_enabled() || !self.requires_grad() {
            return Ok(Tensor::new(out_data, false));
        }

        let softmax_data = a_data.softmax(axis)?;
        let backward_fn: BackwardFn = Arc::new(move |grad| {
            // d = grad - softmax * sum(grad, axis, keepdim=true)
            let sum_grad = grad.sum(axis, true).unwrap();
            let sub_term = softmax_data.mul(&sum_grad).unwrap();
            let d = grad.sub(&sub_term).unwrap();
            vec![Some(d)]
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        })
    }

    /// Numerically stable Softmax along an axis.
    pub fn softmax(&self, axis: usize) -> Result<Tensor> {
        let a_data = self.data();
        let out_data = a_data.softmax(axis)?;

        if !is_grad_enabled() || !self.requires_grad() {
            return Ok(Tensor::new(out_data, false));
        }

        let saved_out = out_data.clone();
        let backward_fn: BackwardFn = Arc::new(move |grad| {
            // d = s * (grad - sum(grad * s, axis, keepdim=true))
            let grad_s = grad.mul(&saved_out).unwrap();
            let sum_grad_s = grad_s.sum(axis, true).unwrap();
            let sub = grad.sub(&sum_grad_s).unwrap();
            let d = saved_out.mul(&sub).unwrap();
            vec![Some(d)]
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        })
    }

    /// Reshape tensor.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Tensor> {
        let a_data = self.data();
        let orig_shape = a_data.shape().to_vec();
        let out_data = a_data.reshape(new_shape)?;

        if !is_grad_enabled() || !self.requires_grad() {
            return Ok(Tensor::new(out_data, false));
        }

        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let d = grad.reshape(&orig_shape).unwrap();
            vec![Some(d)]
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        })
    }

    /// Transpose two dimensions.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Tensor> {
        let a_data = self.data();
        let out_data = a_data.transpose(dim0, dim1)?;

        if !is_grad_enabled() || !self.requires_grad() {
            return Ok(Tensor::new(out_data, false));
        }

        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let d = grad.transpose(dim0, dim1).unwrap();
            vec![Some(d)]
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        })
    }

    /// Permute dimensions.
    pub fn permute(&self, dims: &[usize]) -> Result<Tensor> {
        let a_data = self.data();
        let out_data = a_data.permute(dims)?;

        if !is_grad_enabled() || !self.requires_grad() {
            return Ok(Tensor::new(out_data, false));
        }

        let mut inv_dims = vec![0; dims.len()];
        for (i, &d) in dims.iter().enumerate() {
            inv_dims[d] = i;
        }

        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let d = grad.permute(&inv_dims).unwrap();
            vec![Some(d)]
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        })
    }

    /// Absolute value elementwise.
    pub fn abs(&self) -> Result<Tensor> {
        let a_data = self.data();
        let out_data = a_data.abs()?;

        if !is_grad_enabled() || !self.requires_grad() {
            return Ok(Tensor::new(out_data, false));
        }

        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let sign = a_data.unary_op(|x| {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            });
            let d = grad.mul(&sign).unwrap();
            vec![Some(d)]
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        })
    }

    /// Slices the tensor along an axis from start to end (end-exclusive).
    pub fn slice(&self, axis: usize, start: usize, end: usize) -> Result<Tensor> {
        let a_data = self.data();
        let in_shape = a_data.shape().to_vec();
        let out_data = a_data.slice(axis, start, end)?;

        if !is_grad_enabled() || !self.requires_grad() {
            return Ok(Tensor::new(out_data, false));
        }

        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let ndim = in_shape.len();
            let mut full_grad_data = vec![0.0; numel(&in_shape)];
            let full_strides = compute_c_contiguous_strides(&in_shape);

            let g_contig = grad.to_contiguous();
            let g_shape = g_contig.shape();
            let g_numel = g_contig.numel();

            for idx in 0..g_numel {
                let mut multi = vec![0; ndim];
                flat_to_multi_index(idx, g_shape, &mut multi);
                let val = g_contig.get(&multi);

                multi[axis] += start;
                let full_off = multi_index_to_offset(&multi, &full_strides, 0);
                full_grad_data[full_off] = val;
            }

            vec![Some(RawTensor::from_vec(full_grad_data, in_shape.clone()))]
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        })
    }

    /// Insert a dimension of size 1 at the specified axis.
    pub fn unsqueeze(&self, axis: usize) -> Result<Tensor> {
        let a_data = self.data();
        let orig_shape = a_data.shape().to_vec();
        let out_data = a_data.unsqueeze(axis)?;

        if !is_grad_enabled() || !self.requires_grad() {
            return Ok(Tensor::new(out_data, false));
        }

        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let d = grad.reshape(&orig_shape).unwrap();
            vec![Some(d)]
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        })
    }

    /// Remove a dimension of size 1 at the specified axis.
    pub fn squeeze(&self, axis: usize) -> Result<Tensor> {
        let a_data = self.data();
        let orig_shape = a_data.shape().to_vec();
        let out_data = a_data.squeeze(Some(axis))?;

        if !is_grad_enabled() || !self.requires_grad() {
            return Ok(Tensor::new(out_data, false));
        }

        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let d = grad.reshape(&orig_shape).unwrap();
            vec![Some(d)]
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        })
    }

    /// Concatenates a slice of tensors along an axis with full autograd tracking.
    pub fn cat(tensors: &[&Tensor], axis: usize) -> Result<Tensor> {
        let raw_tensors: Vec<RawTensor> = tensors.iter().map(|t| t.data()).collect();
        let raw_refs: Vec<&RawTensor> = raw_tensors.iter().collect();
        let out_data = RawTensor::cat(&raw_refs, axis)?;

        let has_grad = is_grad_enabled() && tensors.iter().any(|t| t.requires_grad());
        if !has_grad {
            return Ok(Tensor::new(out_data, false));
        }

        let parents: Vec<Arc<TensorInner>> = tensors.iter().map(|t| t.inner.clone()).collect();
        let reqs: Vec<bool> = tensors.iter().map(|t| t.requires_grad()).collect();
        let slice_lengths: Vec<usize> = tensors.iter().map(|t| t.shape()[axis]).collect();

        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let mut grads = Vec::with_capacity(reqs.len());
            let mut start = 0;
            for (&req, &len) in reqs.iter().zip(slice_lengths.iter()) {
                let end = start + len;
                if req {
                    let d = grad.slice(axis, start, end).unwrap();
                    grads.push(Some(d));
                } else {
                    grads.push(None);
                }
                start = end;
            }
            grads
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(out_data, parents, backward_fn),
        })
    }

    /// Flatten tensor to 1D.
    pub fn flatten(&self) -> Result<Tensor> {
        self.reshape(&[self.numel()])
    }

    /// 2D Convolution forward with full autograd tracking.
    pub fn conv2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        params: Conv2dParams,
    ) -> Result<Tensor> {
        let in_data = self.data();
        let w_data = weight.data();
        let b_data = bias.map(|b| b.data());

        let out_data = conv2d_forward(&in_data, &w_data, b_data.as_ref(), params)?;

        let has_grad = self.requires_grad()
            || weight.requires_grad()
            || bias.is_some_and(|b| b.requires_grad());

        if !is_grad_enabled() || !has_grad {
            return Ok(Tensor::new(out_data, false));
        }

        let mut parents = vec![self.inner.clone(), weight.inner.clone()];
        if let Some(b) = bias {
            parents.push(b.inner.clone());
        }

        let in_req = self.requires_grad();
        let w_req = weight.requires_grad();
        let b_req = bias.is_some_and(|b| b.requires_grad());

        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let (d_in, d_w, d_b) = conv2d_backward(grad, &in_data, &w_data, params).unwrap();
            let mut grads = Vec::new();
            grads.push(if in_req { Some(d_in) } else { None });
            grads.push(if w_req { Some(d_w) } else { None });
            if b_req {
                grads.push(d_b);
            }
            grads
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(out_data, parents, backward_fn),
        })
    }

    /// 2D Max pooling forward with full autograd tracking.
    pub fn max_pool2d(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Tensor> {
        let in_data = self.data();
        let (out_data, argmax) = max_pool2d_forward(&in_data, kernel_size, stride)?;

        if !is_grad_enabled() || !self.requires_grad() {
            return Ok(Tensor::new(out_data, false));
        }

        let in_shape = in_data.shape().to_vec();
        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let d_in = max_pool2d_backward(grad, &in_shape, &argmax).unwrap();
            vec![Some(d_in)]
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        })
    }

    /// Embedding table lookup with backward sparse-to-dense gradient scatter.
    pub fn embedding(&self, indices: &[usize]) -> Result<Tensor> {
        let w_data = self.data();
        let out_data = w_data.embedding_lookup(indices)?;

        if !is_grad_enabled() || !self.requires_grad() {
            return Ok(Tensor::new(out_data, false));
        }

        let w_shape = w_data.shape().to_vec();
        let idx_vec = indices.to_vec();

        let backward_fn: BackwardFn = Arc::new(move |grad| {
            let embedding_dim = w_shape[1];
            let mut grad_w_data = vec![0.0; w_shape[0] * embedding_dim];
            let grad_contig = grad.to_contiguous();
            let g_slice = grad_contig.as_slice();

            for (i, &idx) in idx_vec.iter().enumerate() {
                let g_start = i * embedding_dim;
                let w_start = idx * embedding_dim;
                for d in 0..embedding_dim {
                    grad_w_data[w_start + d] += g_slice[g_start + d];
                }
            }

            vec![Some(RawTensor::from_vec(grad_w_data, w_shape.clone()))]
        });

        Ok(Tensor {
            inner: TensorInner::with_parents(out_data, vec![self.inner.clone()], backward_fn),
        })
    }

    /// Returns top-k values and their row-local indices along the last dimension.
    /// Gradient uses straight-through estimator: grad flows only to selected positions.
    /// Returns `(values_tensor, indices_vec)` where indices are local per-row.
    pub fn topk(&self, k: usize) -> Result<(Tensor, Vec<usize>)> {
        let data = self.data();
        let (top_vals_raw, indices) = data.topk(k)?;
        let full_d = data.shape()[data.shape().len() - 1];

        if !is_grad_enabled() || !self.requires_grad() {
            return Ok((Tensor::new(top_vals_raw, false), indices));
        }

        let self_req = self.requires_grad();
        let indices_clone = indices.clone();

        let backward_fn: BackwardFn = Arc::new(move |grad| {
            if !self_req {
                return vec![None];
            }
            let scattered = RawTensor::topk_scatter_back(grad, &indices_clone, full_d).unwrap();
            vec![Some(scattered)]
        });

        let out = Tensor {
            inner: TensorInner::with_parents(top_vals_raw, vec![self.inner.clone()], backward_fn),
        };
        Ok((out, indices))
    }
}

// --- Operator Overloads ---

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Self::Output {
        Tensor::add(self, rhs).expect("Tensor addition failed")
    }
}

impl Add<Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        Tensor::add(self, &rhs).expect("Tensor addition failed")
    }
}

impl Add<&Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Self::Output {
        Tensor::add(&self, rhs).expect("Tensor addition failed")
    }
}

impl Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        Tensor::add(&self, &rhs).expect("Tensor addition failed")
    }
}

impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Self::Output {
        Tensor::sub(self, rhs).expect("Tensor subtraction failed")
    }
}

impl Sub<Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Self::Output {
        Tensor::sub(self, &rhs).expect("Tensor subtraction failed")
    }
}

impl Sub<&Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Self::Output {
        Tensor::sub(&self, rhs).expect("Tensor subtraction failed")
    }
}

impl Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Self::Output {
        Tensor::sub(&self, &rhs).expect("Tensor subtraction failed")
    }
}

impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Self::Output {
        Tensor::mul(self, rhs).expect("Tensor multiplication failed")
    }
}

impl Mul<Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Self::Output {
        Tensor::mul(self, &rhs).expect("Tensor multiplication failed")
    }
}

impl Mul<&Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Self::Output {
        Tensor::mul(&self, rhs).expect("Tensor multiplication failed")
    }
}

impl Mul<Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Self::Output {
        Tensor::mul(&self, &rhs).expect("Tensor multiplication failed")
    }
}

impl Div<&Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Self::Output {
        Tensor::div(self, rhs).expect("Tensor division failed")
    }
}

impl Div<Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Self::Output {
        Tensor::div(self, &rhs).expect("Tensor division failed")
    }
}

impl Div<&Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Self::Output {
        Tensor::div(&self, rhs).expect("Tensor division failed")
    }
}

impl Div<Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Self::Output {
        Tensor::div(&self, &rhs).expect("Tensor division failed")
    }
}

impl Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        Tensor::neg(self)
    }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        Tensor::neg(&self)
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(data={}, requires_grad={})",
            self.data(),
            self.requires_grad()
        )
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.data())
    }
}
