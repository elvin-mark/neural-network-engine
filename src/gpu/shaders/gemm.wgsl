@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

struct Dims {
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
};
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.y;
    let col = global_id.x;
    var sum: f32 = 0.0;

    let num_tiles = (dims.K + 15u) / 16u;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * 16u + local_id.x;
        if (row < dims.M && a_col < dims.K) {
            tile_a[local_id.y][local_id.x] = A[row * dims.K + a_col];
        } else {
            tile_a[local_id.y][local_id.x] = 0.0;
        }

        let b_row = t * 16u + local_id.y;
        if (b_row < dims.K && col < dims.N) {
            tile_b[local_id.y][local_id.x] = B[b_row * dims.N + col];
        } else {
            tile_b[local_id.y][local_id.x] = 0.0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            sum = sum + tile_a[local_id.y][k] * tile_b[k][local_id.x];
        }

        workgroupBarrier();
    }

    if (row < dims.M && col < dims.N) {
        C[row * dims.N + col] = sum;
    }
}
