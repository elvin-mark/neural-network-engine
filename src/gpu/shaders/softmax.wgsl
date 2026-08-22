@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> C: array<f32>;

struct SoftmaxParams {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
};
@group(0) @binding(2) var<uniform> params: SoftmaxParams;

@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.rows) {
        return;
    }

    let offset = row * params.cols;

    // 1. Find row maximum for numerical stability
    var max_val: f32 = -1e30;
    for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
        max_val = max(max_val, A[offset + c]);
    }

    // 2. Compute sum of exponentials
    var sum_exp: f32 = 0.0;
    for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
        sum_exp = sum_exp + exp(A[offset + c] - max_val);
    }

    let inv_sum = 1.0 / max(sum_exp, 1e-9);

    // 3. Normalize probability distribution
    for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
        C[offset + c] = exp(A[offset + c] - max_val) * inv_sum;
    }
}
