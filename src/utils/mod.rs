//! Numerical verification utilities and dataset loaders.

pub mod data;
pub mod gradcheck;

pub use data::{
    generate_digits_dataset, generate_spiral_dataset, generate_xor_dataset, load_iris_dataset,
    standardize, train_test_split, DataLoader, TensorDataset,
};
pub use gradcheck::gradcheck;
