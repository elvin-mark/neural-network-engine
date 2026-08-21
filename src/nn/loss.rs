//! Loss functions (MSELoss, CrossEntropyLoss, BCEWithLogitsLoss, L1Loss).

use crate::autograd::Tensor;
use crate::error::{EngineError, Result};
use crate::tensor::RawTensor;

/// Mean Squared Error loss.
pub struct MSELoss;

impl MSELoss {
    /// Computes MSE loss between predictions and targets: `(1/N) * sum((pred - target)^2)`.
    pub fn forward(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        let diff = pred.sub(target)?;
        let sq = diff.powf(2.0)?;
        Ok(sq.mean_all())
    }
}

/// Numerically stable Cross Entropy loss.
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    /// Computes Cross Entropy loss from raw logits `[N, C]` and target class indices `[N]`.
    pub fn forward_with_indices(logits: &Tensor, target_indices: &[usize]) -> Result<Tensor> {
        let shape = logits.shape();
        if shape.len() != 2 {
            return Err(EngineError::InvalidArgument(format!(
                "CrossEntropyLoss expects 2D logits [BatchSize, NumClasses], got shape {:?}",
                shape
            )));
        }

        let batch_size = shape[0];
        let num_classes = shape[1];

        if batch_size != target_indices.len() {
            return Err(EngineError::ShapeMismatch {
                expected: vec![batch_size],
                actual: vec![target_indices.len()],
            });
        }

        for (b, &target_class) in target_indices.iter().enumerate() {
            if target_class >= num_classes {
                return Err(EngineError::InvalidArgument(format!(
                    "Target class index {} at sample {} is out of bounds for num_classes {}",
                    target_class, b, num_classes
                )));
            }
        }

        let log_probs = logits.log_softmax(1)?;

        // Create one-hot mask for backward propagation
        let mut target_mask = vec![0.0; batch_size * num_classes];
        for (b, &target_class) in target_indices.iter().enumerate() {
            target_mask[b * num_classes + target_class] = 1.0;
        }

        let target_tensor = Tensor::new(
            RawTensor::from_vec(target_mask, vec![batch_size, num_classes]),
            false,
        );

        let nll = log_probs.mul(&target_tensor)?.neg();
        let loss = nll.sum(1, false)?.mean_all();
        Ok(loss)
    }

    /// Computes Cross Entropy loss from raw logits `[N, C]` and target probabilities `[N, C]`.
    pub fn forward_with_probabilities(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let l_shape = logits.shape();
        let t_shape = targets.shape();
        if l_shape.len() != 2 {
            return Err(EngineError::InvalidArgument(format!(
                "CrossEntropyLoss expects 2D logits [BatchSize, NumClasses], got rank {} with shape {:?}",
                l_shape.len(),
                l_shape
            )));
        }
        if t_shape.len() != 2 {
            return Err(EngineError::InvalidArgument(format!(
                "CrossEntropyLoss expects 2D targets [BatchSize, NumClasses], got rank {} with shape {:?}",
                t_shape.len(),
                t_shape
            )));
        }
        if l_shape != t_shape {
            return Err(EngineError::ShapeMismatch {
                expected: l_shape,
                actual: t_shape,
            });
        }

        let log_probs = logits.log_softmax(1)?;
        let nll = log_probs.mul(targets)?.neg();
        let loss = nll.sum(1, false)?.mean_all();
        Ok(loss)
    }
}

/// Binary Cross Entropy with Logits loss: max(x, 0) - x * y + log(1 + exp(-|x|)).
pub struct BCEWithLogitsLoss;

impl BCEWithLogitsLoss {
    pub fn forward(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // max(x, 0)
        let relu_x = logits.relu()?;
        let x_mul_y = logits.mul(targets)?;

        // -|x|
        let neg_abs_x = logits.abs()?.neg();
        let exp_neg_abs = neg_abs_x.exp()?;
        let one_plus_exp = exp_neg_abs.add(&Tensor::scalar(1.0, false))?;
        let log_term = one_plus_exp.log()?;

        let loss = relu_x.sub(&x_mul_y)?.add(&log_term)?;
        Ok(loss.mean_all())
    }
}

/// L1 / Mean Absolute Error loss.
pub struct L1Loss;

impl L1Loss {
    pub fn forward(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        let diff = pred.sub(target)?;
        let abs_diff = diff.abs()?;
        Ok(abs_diff.mean_all())
    }
}
