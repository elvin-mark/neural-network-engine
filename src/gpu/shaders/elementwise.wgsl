@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

struct OpParams {
    len: u32,
    op: u32,
    scalar: f32,
    b_len: u32,
};
@group(0) @binding(3) var<uniform> params: OpParams;

const SQRT_2_OVER_PI: f32 = 0.7978845608;
const GELU_COEF: f32 = 0.044715;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.len) {
        return;
    }

    let a_val = A[idx];
    let b_idx = idx % max(params.b_len, 1u);
    let b_val = B[b_idx];

    switch (params.op) {
        // 0: Add
        case 0u: {
            C[idx] = a_val + b_val;
        }
        // 1: Sub
        case 1u: {
            C[idx] = a_val - b_val;
        }
        // 2: Mul
        case 2u: {
            C[idx] = a_val * b_val;
        }
        // 3: Div
        case 3u: {
            C[idx] = a_val / b_val;
        }
        // 4: ReLU
        case 4u: {
            C[idx] = max(0.0, a_val);
        }
        // 5: GELU (Approximation)
        case 5u: {
            let cube = a_val * a_val * a_val;
            let inner = SQRT_2_OVER_PI * (a_val + GELU_COEF * cube);
            C[idx] = 0.5 * a_val * (1.0 + tanh(inner));
        }
        // 6: SiLU (Swish)
        case 6u: {
            let sig = 1.0 / (1.0 + exp(-a_val));
            C[idx] = a_val * sig;
        }
        // 7: Tanh
        case 7u: {
            C[idx] = tanh(a_val);
        }
        // 8: Sigmoid
        case 8u: {
            C[idx] = 1.0 / (1.0 + exp(-a_val));
        }
        // 9: Scale
        case 9u: {
            C[idx] = a_val * params.scalar;
        }
        default: {
            C[idx] = a_val;
        }
    }
}
