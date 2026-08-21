//! Optimization algorithms (SGD, Adam, AdamW, RMSprop).

pub mod adam;
pub mod rmsprop;
pub mod sgd;

pub use adam::Adam;
pub use rmsprop::RMSprop;
pub use sgd::SGD;
