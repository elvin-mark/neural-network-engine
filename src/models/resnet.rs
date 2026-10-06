//! Deep Residual Networks (ResNet-18, ResNet-34, ResNet-50) with Residual and Bottleneck skip connections.
//!
//! Features:
//! - [`ResidualBlock`]: Basic residual block with 2x 3x3 Conv2d and identity/downsampling shortcut.
//! - [`BottleneckBlock`]: 3-layer bottleneck block (1x1 -> 3x3 -> 1x1 Conv2d) with 4x channel expansion.
//! - [`ResNet`]: Full configurable architecture supporting both standard ImageNet and CIFAR-10/100 stems.
//! - Predefined constructors: [`ResNet::resnet18`], [`ResNet::resnet34`], [`ResNet::resnet50`], [`ResNet::cifar_resnet18`].

use crate::autograd::Tensor;
use crate::error::Result;
use crate::nn::conv::Conv2d;
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::nn::norm::BatchNorm2d;
use crate::nn::pooling::MaxPool2d;

// =========================================================================
// 1. RESIDUAL & BOTTLENECK BLOCKS
// =========================================================================

/// Basic Residual Block for ResNet-18 and ResNet-34:
/// $$y = \text{ReLU}(\text{BN}(\text{Conv}_{3\times3}(\text{ReLU}(\text{BN}(\text{Conv}_{3\times3}(x))))) + \text{shortcut}(x))$$
#[derive(Clone)]
pub struct ResidualBlock {
    pub conv1: Conv2d,
    pub bn1: BatchNorm2d,
    pub conv2: Conv2d,
    pub bn2: BatchNorm2d,
    pub downsample: Option<(Conv2d, BatchNorm2d)>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub stride: usize,
}

impl ResidualBlock {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        let conv1 = Conv2d::with_options(
            in_channels,
            out_channels,
            (3, 3),
            (stride, stride),
            (1, 1),
            (1, 1),
            false,
        );
        let bn1 = BatchNorm2d::new(out_channels);

        let conv2 = Conv2d::with_options(
            out_channels,
            out_channels,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            false,
        );
        let bn2 = BatchNorm2d::new(out_channels);

        let downsample = if stride != 1 || in_channels != out_channels {
            let ds_conv = Conv2d::with_options(
                in_channels,
                out_channels,
                (1, 1),
                (stride, stride),
                (0, 0),
                (1, 1),
                false,
            );
            let ds_bn = BatchNorm2d::new(out_channels);
            Some((ds_conv, ds_bn))
        } else {
            None
        };

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            downsample,
            in_channels,
            out_channels,
            stride,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = x.clone();

        let mut out = self.conv1.forward(x)?;
        out = self.bn1.forward(&out)?.relu()?;

        out = self.conv2.forward(&out)?;
        out = self.bn2.forward(&out)?;

        let shortcut = if let Some((ref ds_conv, ref ds_bn)) = self.downsample {
            ds_bn.forward(&ds_conv.forward(&identity)?)?
        } else {
            identity
        };

        out.add(&shortcut)?.relu()
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        if let Some((ref ds_conv, ref ds_bn)) = self.downsample {
            params.extend(ds_conv.parameters());
            params.extend(ds_bn.parameters());
        }
        params
    }

    pub fn train(&mut self) {
        self.bn1.train();
        self.bn2.train();
        if let Some((_, ref mut ds_bn)) = self.downsample {
            ds_bn.train();
        }
    }

    pub fn eval(&mut self) {
        self.bn1.eval();
        self.bn2.eval();
        if let Some((_, ref mut ds_bn)) = self.downsample {
            ds_bn.eval();
        }
    }
}

/// Bottleneck Block for deep ResNets (ResNet-50, ResNet-101, ResNet-152):
/// 1x1 Conv (reduce) -> 3x3 Conv -> 1x1 Conv (expand by 4x).
#[derive(Clone)]
pub struct BottleneckBlock {
    pub conv1: Conv2d,
    pub bn1: BatchNorm2d,
    pub conv2: Conv2d,
    pub bn2: BatchNorm2d,
    pub conv3: Conv2d,
    pub bn3: BatchNorm2d,
    pub downsample: Option<(Conv2d, BatchNorm2d)>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub expansion: usize,
    pub stride: usize,
}

impl BottleneckBlock {
    pub const EXPANSION: usize = 4;

    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        let expansion = Self::EXPANSION;
        let expanded_channels = out_channels * expansion;

        // 1x1 Conv (channel reduction)
        let conv1 = Conv2d::with_options(
            in_channels,
            out_channels,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            false,
        );
        let bn1 = BatchNorm2d::new(out_channels);

        // 3x3 Conv (spatial convolution)
        let conv2 = Conv2d::with_options(
            out_channels,
            out_channels,
            (3, 3),
            (stride, stride),
            (1, 1),
            (1, 1),
            false,
        );
        let bn2 = BatchNorm2d::new(out_channels);

        // 1x1 Conv (channel expansion)
        let conv3 = Conv2d::with_options(
            out_channels,
            expanded_channels,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            false,
        );
        let bn3 = BatchNorm2d::new(expanded_channels);

        let downsample = if stride != 1 || in_channels != expanded_channels {
            let ds_conv = Conv2d::with_options(
                in_channels,
                expanded_channels,
                (1, 1),
                (stride, stride),
                (0, 0),
                (1, 1),
                false,
            );
            let ds_bn = BatchNorm2d::new(expanded_channels);
            Some((ds_conv, ds_bn))
        } else {
            None
        };

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            downsample,
            in_channels,
            out_channels,
            expansion,
            stride,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = x.clone();

        let mut out = self.conv1.forward(x)?;
        out = self.bn1.forward(&out)?.relu()?;

        out = self.conv2.forward(&out)?;
        out = self.bn2.forward(&out)?.relu()?;

        out = self.conv3.forward(&out)?;
        out = self.bn3.forward(&out)?;

        let shortcut = if let Some((ref ds_conv, ref ds_bn)) = self.downsample {
            ds_bn.forward(&ds_conv.forward(&identity)?)?
        } else {
            identity
        };

        out.add(&shortcut)?.relu()
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        params.extend(self.conv3.parameters());
        params.extend(self.bn3.parameters());
        if let Some((ref ds_conv, ref ds_bn)) = self.downsample {
            params.extend(ds_conv.parameters());
            params.extend(ds_bn.parameters());
        }
        params
    }

    pub fn train(&mut self) {
        self.bn1.train();
        self.bn2.train();
        self.bn3.train();
        if let Some((_, ref mut ds_bn)) = self.downsample {
            ds_bn.train();
        }
    }

    pub fn eval(&mut self) {
        self.bn1.eval();
        self.bn2.eval();
        self.bn3.eval();
        if let Some((_, ref mut ds_bn)) = self.downsample {
            ds_bn.eval();
        }
    }
}

// =========================================================================
// 2. ENUM WRAPPER FOR RESIDUAL LAYERS
// =========================================================================

#[derive(Clone)]
pub enum ResBlock {
    Basic(ResidualBlock),
    Bottleneck(BottleneckBlock),
}

impl ResBlock {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            ResBlock::Basic(b) => b.forward(x),
            ResBlock::Bottleneck(b) => b.forward(x),
        }
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        match self {
            ResBlock::Basic(b) => b.parameters(),
            ResBlock::Bottleneck(b) => b.parameters(),
        }
    }

    pub fn train(&mut self) {
        match self {
            ResBlock::Basic(b) => b.train(),
            ResBlock::Bottleneck(b) => b.train(),
        }
    }

    pub fn eval(&mut self) {
        match self {
            ResBlock::Basic(b) => b.eval(),
            ResBlock::Bottleneck(b) => b.eval(),
        }
    }
}

// =========================================================================
// 3. FULL RESNET ARCHITECTURE
// =========================================================================

/// Complete Residual Network architecture.
#[derive(Clone)]
pub struct ResNet {
    pub conv1: Conv2d,
    pub bn1: BatchNorm2d,
    pub maxpool: Option<MaxPool2d>,
    pub layer1: Vec<ResBlock>,
    pub layer2: Vec<ResBlock>,
    pub layer3: Vec<ResBlock>,
    pub layer4: Vec<ResBlock>,
    pub fc: Linear,
    pub in_channels: usize,
    pub num_classes: usize,
}

impl ResNet {
    /// Creates a ResNet-18 model for standard ImageNet images (3 channels, 7x7 stem with MaxPool).
    pub fn resnet18(in_channels: usize, num_classes: usize) -> Self {
        Self::build(in_channels, num_classes, &[2, 2, 2, 2], false, true)
    }

    /// Creates a ResNet-18 model tailored for 32x32 CIFAR-10 / CIFAR-100 images (3x3 stem, no MaxPool).
    pub fn cifar_resnet18(in_channels: usize, num_classes: usize) -> Self {
        Self::build(in_channels, num_classes, &[2, 2, 2, 2], false, false)
    }

    /// Creates a ResNet-34 model.
    pub fn resnet34(in_channels: usize, num_classes: usize) -> Self {
        Self::build(in_channels, num_classes, &[3, 4, 6, 3], false, true)
    }

    /// Creates a ResNet-50 model with Bottleneck blocks.
    pub fn resnet50(in_channels: usize, num_classes: usize) -> Self {
        Self::build(in_channels, num_classes, &[3, 4, 6, 3], true, true)
    }

    /// Flexible builder for custom ResNet architectures.
    pub fn build(
        in_channels: usize,
        num_classes: usize,
        layers: &[usize; 4],
        is_bottleneck: bool,
        use_imagenet_stem: bool,
    ) -> Self {
        let base_channels = 64;

        let (conv1, bn1, maxpool) = if use_imagenet_stem {
            // Standard ImageNet 7x7 Conv + MaxPool
            let c1 = Conv2d::with_options(
                in_channels,
                base_channels,
                (7, 7),
                (2, 2),
                (3, 3),
                (1, 1),
                false,
            );
            let b1 = BatchNorm2d::new(base_channels);
            let mp = Some(MaxPool2d::new((3, 3), (2, 2)));
            (c1, b1, mp)
        } else {
            // CIFAR 3x3 Conv without MaxPool
            let c1 = Conv2d::with_options(
                in_channels,
                base_channels,
                (3, 3),
                (1, 1),
                (1, 1),
                (1, 1),
                false,
            );
            let b1 = BatchNorm2d::new(base_channels);
            (c1, b1, None)
        };

        let mut current_channels = base_channels;

        let layer1 = Self::make_layer(&mut current_channels, 64, layers[0], 1, is_bottleneck);
        let layer2 = Self::make_layer(&mut current_channels, 128, layers[1], 2, is_bottleneck);
        let layer3 = Self::make_layer(&mut current_channels, 256, layers[2], 2, is_bottleneck);
        let layer4 = Self::make_layer(&mut current_channels, 512, layers[3], 2, is_bottleneck);

        let final_features = current_channels;
        let fc = Linear::new(final_features, num_classes);

        Self {
            conv1,
            bn1,
            maxpool,
            layer1,
            layer2,
            layer3,
            layer4,
            fc,
            in_channels,
            num_classes,
        }
    }

    fn make_layer(
        current_channels: &mut usize,
        out_channels: usize,
        num_blocks: usize,
        stride: usize,
        is_bottleneck: bool,
    ) -> Vec<ResBlock> {
        let mut blocks = Vec::with_capacity(num_blocks);

        if is_bottleneck {
            blocks.push(ResBlock::Bottleneck(BottleneckBlock::new(
                *current_channels,
                out_channels,
                stride,
            )));
            *current_channels = out_channels * BottleneckBlock::EXPANSION;

            for _ in 1..num_blocks {
                blocks.push(ResBlock::Bottleneck(BottleneckBlock::new(
                    *current_channels,
                    out_channels,
                    1,
                )));
            }
        } else {
            blocks.push(ResBlock::Basic(ResidualBlock::new(
                *current_channels,
                out_channels,
                stride,
            )));
            *current_channels = out_channels;

            for _ in 1..num_blocks {
                blocks.push(ResBlock::Basic(ResidualBlock::new(
                    *current_channels,
                    out_channels,
                    1,
                )));
            }
        }

        blocks
    }

    /// Computes the forward pass on 4D image batch [BatchSize, InChannels, Height, Width].
    /// Returns 2D classification logits [BatchSize, NumClasses].
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 1. Stem
        let mut out = self.conv1.forward(x)?;
        out = self.bn1.forward(&out)?.relu()?;

        if let Some(ref mp) = self.maxpool {
            out = mp.forward(&out)?;
        }

        // 2. Residual stages
        for block in &self.layer1 {
            out = block.forward(&out)?;
        }
        for block in &self.layer2 {
            out = block.forward(&out)?;
        }
        for block in &self.layer3 {
            out = block.forward(&out)?;
        }
        for block in &self.layer4 {
            out = block.forward(&out)?;
        }

        // 3. Global Average Pooling: [B, C, H, W] -> [B, C]
        let pooled = out.mean(3, true)?.mean(2, true)?.squeeze(3)?.squeeze(2)?;

        // 4. Fully Connected Head
        self.fc.forward(&pooled)
    }
}

impl Module for ResNet {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        for block in &self.layer1 {
            params.extend(block.parameters());
        }
        for block in &self.layer2 {
            params.extend(block.parameters());
        }
        for block in &self.layer3 {
            params.extend(block.parameters());
        }
        for block in &self.layer4 {
            params.extend(block.parameters());
        }
        params.extend(self.fc.parameters());
        params
    }

    fn train(&mut self) {
        self.bn1.train();
        for block in &mut self.layer1 {
            block.train();
        }
        for block in &mut self.layer2 {
            block.train();
        }
        for block in &mut self.layer3 {
            block.train();
        }
        for block in &mut self.layer4 {
            block.train();
        }
    }

    fn eval(&mut self) {
        self.bn1.eval();
        for block in &mut self.layer1 {
            block.eval();
        }
        for block in &mut self.layer2 {
            block.eval();
        }
        for block in &mut self.layer3 {
            block.eval();
        }
        for block in &mut self.layer4 {
            block.eval();
        }
    }
}
