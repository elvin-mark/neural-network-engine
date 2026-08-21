//! Numerical verification utilities and dataset loaders.

pub mod data;
pub mod gradcheck;

pub use data::{generate_spiral_dataset, generate_xor_dataset, DataLoader, TensorDataset};
pub use gradcheck::gradcheck;
