# Dataset Utilities & Computer Vision Transforms

Data loading pipelines, batching, shuffling, dataset loaders (MNIST, CIFAR, Iris), and vision augmentations.

**Source files**:
- Data Loading: [`src/utils/data.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/utils/data.rs)
- Vision Transforms: [`src/vision/transforms.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/vision/transforms.rs)
- Download Utilities: [`scripts/download_datasets.sh`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/scripts/download_datasets.sh)

---

## Data Loading Subsystem (`DataLoader`)

### 1. `TensorDataset`
Holds paired inputs and labels:
```rust
let dataset = TensorDataset::new(x_tensor, y_tensor);
```

### 2. `DataLoader`
Batches and optionally shuffles datasets across epochs:
```rust
let mut dataloader = DataLoader::new(dataset, batch_size: 64, shuffle: true);

for (batch_x, batch_y) in dataloader.iter() {
    // Training step
}
```

### 3. Built-in Dataset Readers
- **MNIST**: Reads IDX format binary files (`load_mnist_from_idx`).
- **CIFAR-10 & CIFAR-100**: Reads raw binary batch files (`load_cifar10_from_binary`, `load_cifar100_from_binary`).
- **Fisher's Iris & Digits**: CSV readers (`load_iris_from_csv`, `load_digits_from_csv`).
- **TinyStories**: Autoregressive language modeling text chunker (`load_tinystories_dataset`).

---

## Computer Vision Transforms (`vision::transforms`)

Implements data augmentation over 4D image tensors $[N, C, H, W]$:

```mermaid
flowchart LR
    Raw["Raw Image: [3, 32, 32]"] --> HFlip["RandomHorizontalFlip(p = 0.5)"]
    HFlip --> Crop["RandomCrop((32, 32), padding = 4)"]
    Crop --> Norm["Normalize(mean, std)"]
    Norm --> Out["Augmented Batch"]
```

| Transform | Method / Configuration | Description |
|---|---|---|
| **`RandomHorizontalFlip`** | `p: f32` | Flips image along horizontal axis with probability $p$. |
| **`RandomVerticalFlip`** | `p: f32` | Flips image vertically with probability $p$. |
| **`RandomRotation90`** | None | Rotates image randomly by $0^\circ, 90^\circ, 180^\circ,$ or $270^\circ$. |
| **`RandomCrop`** | `size: (H, W), padding: usize` | Reflects borders and extracts a random spatial crop. |
| **`ColorJitter`** | `brightness, contrast` | Perturbs pixel intensities and dynamic range. |
| **`Normalize`** | `mean: Vec<f32>, std: Vec<f32>` | Channel-wise standardization: $x_c \leftarrow \frac{x_c - \mu_c}{\sigma_c}$. |
| **`Compose`** | `Vec<Box<dyn Transform>>` | Chains multiple transforms sequentially. |

---

## See Also
- [ResNet Architecture](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/models/resnet.md)
- [Vision Transformer](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/models/vit.md)
