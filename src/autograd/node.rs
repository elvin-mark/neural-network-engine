//! Autograd computation graph node and backward gradient dispatch.

use crate::error::Result;
use crate::tensor::RawTensor;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// Backward gradient computation function.
/// Takes incoming gradient w.r.t the node's output and computes gradients w.r.t each input parent.
pub type BackwardFn = Arc<dyn Fn(&RawTensor) -> Vec<Option<RawTensor>> + Send + Sync>;

/// Reduces a higher-dimensional or broadcasted gradient to match a target tensor shape.
pub fn unbroadcast_to(grad: &RawTensor, target_shape: &[usize]) -> Result<RawTensor> {
    if grad.shape() == target_shape {
        return Ok(grad.clone());
    }

    if target_shape.is_empty() {
        return Ok(RawTensor::scalar(grad.sum_all()));
    }

    let grad_ndim = grad.ndim();
    let target_ndim = target_shape.len();

    let mut current = grad.clone();

    // 1. Sum out extra leading broadcast dimensions
    if grad_ndim > target_ndim {
        let diff = grad_ndim - target_ndim;
        for _ in 0..diff {
            current = current.sum(0, false)?;
        }
    }

    // 2. Sum out dimensions that were expanded from 1
    for (i, &target_dim) in target_shape.iter().enumerate() {
        if target_dim == 1 && current.shape()[i] > 1 {
            current = current.sum(i, true)?;
        }
    }

    current.reshape(target_shape)
}

/// Internal representation of an autograd node.
pub struct TensorInner {
    pub data: RwLock<RawTensor>,
    pub grad: RwLock<Option<RawTensor>>,
    pub parents: Vec<Arc<TensorInner>>,
    pub backward_fn: Option<BackwardFn>,
    pub requires_grad: AtomicBool,
    pub id: usize,
}

static NODE_COUNTER: AtomicUsize = AtomicUsize::new(1);

impl TensorInner {
    pub fn new(data: RawTensor, requires_grad: bool) -> Arc<Self> {
        let id = NODE_COUNTER.fetch_add(1, Ordering::Relaxed);
        Arc::new(Self {
            data: RwLock::new(data),
            grad: RwLock::new(None),
            parents: Vec::new(),
            backward_fn: None,
            requires_grad: AtomicBool::new(requires_grad),
            id,
        })
    }

    pub fn with_parents(
        data: RawTensor,
        parents: Vec<Arc<TensorInner>>,
        backward_fn: BackwardFn,
    ) -> Arc<Self> {
        let id = NODE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let requires_grad = parents
            .iter()
            .any(|p| p.requires_grad.load(Ordering::Relaxed));
        Arc::new(Self {
            data: RwLock::new(data),
            grad: RwLock::new(None),
            parents,
            backward_fn: if requires_grad {
                Some(backward_fn)
            } else {
                None
            },
            requires_grad: AtomicBool::new(requires_grad),
            id,
        })
    }

    pub fn accumulate_grad(&self, incoming: RawTensor) {
        if !self.requires_grad.load(Ordering::Relaxed) {
            return;
        }

        let mut grad_guard = self.grad.write().unwrap();
        if let Some(existing) = grad_guard.as_mut() {
            let combined = existing.add(&incoming).expect("Failed to add gradient");
            *existing = combined;
        } else {
            *grad_guard = Some(incoming);
        }
    }
}
