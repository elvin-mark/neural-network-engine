@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Gamma: array<f32>;
@group(0) @binding(2) var<storage, read> Beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;

struct NormParams {
    rows: u32,
    cols: u32,
    eps: f32,
    is_rmsnorm: u32,
};
@group(0) @binding(4) var<uniform> params: NormParams;

@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.rows) {
        return;
    }

    let offset = row * params.cols;

    if (params.is_rmsnorm == 1u) {
        // RMSNorm: Y = X / sqrt(mean(X^2) + eps) * Gamma
        var sum_sq: f32 = 0.0;
        for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
            let val = X[offset + c];
            sum_sq = sum_sq + val * val;
        }

        let rms = sqrt(sum_sq / f32(params.cols) + params.eps);
        let inv_rms = 1.0 / rms;

        for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
            Y[offset + c] = (X[offset + c] * inv_rms) * Gamma[c];
        }
    } else {
        // Standard LayerNorm: Y = (X - mean) / sqrt(var + eps) * Gamma + Beta
        var sum: f32 = 0.0;
        for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
            sum = sum + X[offset + c];
        }
        let mean = sum / f32(params.cols);

        var var_sum: f32 = 0.0;
        for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
            let diff = X[offset + c] - mean;
            var_sum = var_sum + diff * diff;
        }
        let variance = var_sum / f32(params.cols);
        let inv_std = 1.0 / sqrt(variance + params.eps);

        for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
            Y[offset + c] = ((X[offset + c] - mean) * inv_std) * Gamma[c] + Beta[c];
        }
    }
}
