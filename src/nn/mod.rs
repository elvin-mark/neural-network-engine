//! Composable neural network layers, activation functions, and losses.

pub mod activations;
pub mod conv;
pub mod dropout;
pub mod embedding;
pub mod linear;
pub mod loss;
pub mod module;
pub mod norm;
pub mod pooling;
pub mod sequential;

pub use activations::{LeakyReLU, ReLU, Sigmoid, Softmax, Tanh, GELU};
pub use conv::Conv2d;
pub use dropout::Dropout;
pub use embedding::Embedding;
pub use linear::Linear;
pub use loss::{BCEWithLogitsLoss, CrossEntropyLoss, L1Loss, MSELoss};
pub use module::Module;
pub use norm::{BatchNorm1d, LayerNorm};
pub use pooling::MaxPool2d;
pub use sequential::Sequential;
