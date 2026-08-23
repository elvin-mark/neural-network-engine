//! Learning rate schedulers for dynamic learning rate decay and warmup.
//!
//! Includes:
//! - `StepLR`: Step decay by `gamma` every `step_size` epochs.
//! - `MultiStepLR`: Multi-step decay by `gamma` at specified milestone epochs.
//! - `ExponentialLR`: Exponential decay by `gamma` every epoch ($\text{lr} = \text{base\_lr} \times \gamma^t$).
//! - `CosineAnnealingLR`: Cosine annealing decay from `base_lr` to `eta_min`.
//! - `LinearWarmupCosineLR`: Linear warmup followed by cosine decay (essential for LLMs and Transformers).

use crate::optim::Optimizer;
use std::f32::consts::PI;

/// Trait for learning rate schedulers.
pub trait LRScheduler {
    /// Advances the scheduler by one epoch/step, updates the optimizer's learning rate, and returns the new learning rate.
    fn step(&mut self, optimizer: &mut dyn Optimizer) -> f32;

    /// Returns the last computed learning rate.
    fn get_last_lr(&self) -> f32;

    /// Returns the current epoch/step counter.
    fn get_last_epoch(&self) -> usize;
}

/// Decays the learning rate of each parameter group by `gamma` every `step_size` epochs.
#[derive(Debug, Clone)]
pub struct StepLR {
    pub step_size: usize,
    pub gamma: f32,
    pub last_epoch: usize,
    pub base_lr: f32,
    pub last_lr: f32,
}

impl StepLR {
    /// Creates a new `StepLR` scheduler.
    pub fn new(base_lr: f32, step_size: usize, gamma: f32) -> Self {
        assert!(step_size > 0, "step_size must be greater than 0");
        Self {
            step_size,
            gamma,
            last_epoch: 0,
            base_lr,
            last_lr: base_lr,
        }
    }
}

impl LRScheduler for StepLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) -> f32 {
        self.last_epoch += 1;
        let factor = self.gamma.powi((self.last_epoch / self.step_size) as i32);
        let new_lr = self.base_lr * factor;
        self.last_lr = new_lr;
        optimizer.set_lr(new_lr);
        new_lr
    }

    fn get_last_lr(&self) -> f32 {
        self.last_lr
    }

    fn get_last_epoch(&self) -> usize {
        self.last_epoch
    }
}

/// Decays the learning rate of each parameter group by `gamma` once the number of epochs reaches one of the milestones.
#[derive(Debug, Clone)]
pub struct MultiStepLR {
    pub milestones: Vec<usize>,
    pub gamma: f32,
    pub last_epoch: usize,
    pub base_lr: f32,
    pub last_lr: f32,
}

impl MultiStepLR {
    /// Creates a new `MultiStepLR` scheduler.
    pub fn new(base_lr: f32, mut milestones: Vec<usize>, gamma: f32) -> Self {
        milestones.sort();
        Self {
            milestones,
            gamma,
            last_epoch: 0,
            base_lr,
            last_lr: base_lr,
        }
    }
}

impl LRScheduler for MultiStepLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) -> f32 {
        self.last_epoch += 1;
        let count = self
            .milestones
            .iter()
            .filter(|&&m| self.last_epoch >= m)
            .count();
        let factor = self.gamma.powi(count as i32);
        let new_lr = self.base_lr * factor;
        self.last_lr = new_lr;
        optimizer.set_lr(new_lr);
        new_lr
    }

    fn get_last_lr(&self) -> f32 {
        self.last_lr
    }

    fn get_last_epoch(&self) -> usize {
        self.last_epoch
    }
}

/// Decays the learning rate of each parameter group by `gamma` every epoch ($\text{lr} = \text{base\_lr} \times \gamma^t$).
#[derive(Debug, Clone)]
pub struct ExponentialLR {
    pub gamma: f32,
    pub last_epoch: usize,
    pub base_lr: f32,
    pub last_lr: f32,
}

impl ExponentialLR {
    /// Creates a new `ExponentialLR` scheduler.
    pub fn new(base_lr: f32, gamma: f32) -> Self {
        Self {
            gamma,
            last_epoch: 0,
            base_lr,
            last_lr: base_lr,
        }
    }
}

impl LRScheduler for ExponentialLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) -> f32 {
        self.last_epoch += 1;
        let factor = self.gamma.powi(self.last_epoch as i32);
        let new_lr = self.base_lr * factor;
        self.last_lr = new_lr;
        optimizer.set_lr(new_lr);
        new_lr
    }

    fn get_last_lr(&self) -> f32 {
        self.last_lr
    }

    fn get_last_epoch(&self) -> usize {
        self.last_epoch
    }
}

/// Set the learning rate of each parameter group using a cosine annealing schedule.
///
/// $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}}\pi\right)\right)$$
#[derive(Debug, Clone)]
pub struct CosineAnnealingLR {
    pub t_max: usize,
    pub eta_min: f32,
    pub base_lr: f32,
    pub last_epoch: usize,
    pub last_lr: f32,
}

impl CosineAnnealingLR {
    /// Creates a new `CosineAnnealingLR` scheduler.
    pub fn new(base_lr: f32, t_max: usize, eta_min: f32) -> Self {
        assert!(t_max > 0, "t_max must be greater than 0");
        Self {
            t_max,
            eta_min,
            base_lr,
            last_epoch: 0,
            last_lr: base_lr,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) -> f32 {
        self.last_epoch += 1;
        let t = self.last_epoch.min(self.t_max);
        let cos_val = (PI * (t as f32) / (self.t_max as f32)).cos();
        let new_lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + cos_val);
        self.last_lr = new_lr;
        optimizer.set_lr(new_lr);
        new_lr
    }

    fn get_last_lr(&self) -> f32 {
        self.last_lr
    }

    fn get_last_epoch(&self) -> usize {
        self.last_epoch
    }
}

/// Linear Warmup followed by Cosine Annealing learning rate schedule.
///
/// Standard schedule for training modern Large Language Models (LLaMA, GPT, BERT) and Transformers.
#[derive(Debug, Clone)]
pub struct LinearWarmupCosineLR {
    pub warmup_steps: usize,
    pub max_steps: usize,
    pub base_lr: f32,
    pub min_lr: f32,
    pub warmup_start_lr: f32,
    pub last_step: usize,
    pub last_lr: f32,
}

impl LinearWarmupCosineLR {
    /// Creates a new `LinearWarmupCosineLR` scheduler.
    pub fn new(
        base_lr: f32,
        warmup_steps: usize,
        max_steps: usize,
        min_lr: f32,
        warmup_start_lr: f32,
    ) -> Self {
        assert!(
            max_steps >= warmup_steps,
            "max_steps must be >= warmup_steps"
        );
        Self {
            warmup_steps,
            max_steps,
            base_lr,
            min_lr,
            warmup_start_lr,
            last_step: 0,
            last_lr: warmup_start_lr,
        }
    }
}

impl LRScheduler for LinearWarmupCosineLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) -> f32 {
        self.last_step += 1;
        let new_lr = if self.last_step <= self.warmup_steps {
            if self.warmup_steps == 0 {
                self.base_lr
            } else {
                let progress = self.last_step as f32 / self.warmup_steps as f32;
                self.warmup_start_lr + progress * (self.base_lr - self.warmup_start_lr)
            }
        } else if self.last_step >= self.max_steps {
            self.min_lr
        } else {
            let decay_steps = self.max_steps - self.warmup_steps;
            let current_decay_step = self.last_step - self.warmup_steps;
            let progress = current_decay_step as f32 / decay_steps as f32;
            let cos_val = (PI * progress).cos();
            self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + cos_val)
        };

        self.last_lr = new_lr;
        optimizer.set_lr(new_lr);
        new_lr
    }

    fn get_last_lr(&self) -> f32 {
        self.last_lr
    }

    fn get_last_epoch(&self) -> usize {
        self.last_step
    }
}
