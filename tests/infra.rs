#[cfg(feature = "gpu")]
#[path = "infra/gpu_tests.rs"]
mod gpu_tests;

#[path = "infra/serialization_tests.rs"]
mod serialization_tests;

#[path = "infra/tokenizer_tests.rs"]
mod tokenizer_tests;

#[path = "infra/vision_transforms_tests.rs"]
mod vision_transforms_tests;
