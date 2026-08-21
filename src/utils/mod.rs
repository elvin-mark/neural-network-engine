//! Numerical verification utilities and dataset loaders.

pub mod data;
pub mod gradcheck;

pub use data::{
    generate_cifar100_dataset, generate_cifar10_dataset, generate_digits_dataset,
    generate_mnist_dataset, generate_spiral_dataset, generate_xor_dataset, load_cifar100_dataset,
    load_cifar100_from_binary, load_cifar10_dataset, load_cifar10_from_binary, load_digits_dataset,
    load_digits_from_csv, load_iris_dataset, load_iris_from_csv, load_mnist_dataset,
    load_mnist_from_idx, standardize, train_test_split, DataLoader, TensorDataset, CIFAR10_CLASSES,
};
pub use gradcheck::gradcheck;
