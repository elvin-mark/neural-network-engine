//! Numerical verification utilities and dataset loaders.

pub mod data;
pub mod gradcheck;

pub use data::{
    generate_digits_dataset, generate_spiral_dataset, generate_xor_dataset, load_digits_dataset,
    load_digits_from_csv, load_iris_dataset, load_iris_from_csv, standardize, train_test_split,
    DataLoader, TensorDataset,
};
pub use gradcheck::gradcheck;
