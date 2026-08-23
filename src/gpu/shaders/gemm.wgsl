// High-Performance Vectorized (Vec4) 2D Tiled Matrix Multiplication (GEMM) in WGSL.
//
// Features:
// - 64x64 output block per workgroup with 256 threads (16x16 workgroup size).
// - 4x4 register-tiled outer-product microkernel per thread (16 outputs in registers).
// - Vectorized 128-bit memory loads (4 floats per thread load transaction).
// - 8 KiB shared memory cache-blocking for tile_a [64, 16] and tile_b [16, 64].
// - Full boundary handling for arbitrary non-power-of-two matrix dimensions.

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

// Shared memory tiles: tile_a [64, 16] and tile_b [16, 64]
var<workgroup> tile_a: array<array<f32, 16>, 64>;
var<workgroup> tile_b: array<array<f32, 64>, 16>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let lx = local_id.x;
    let ly = local_id.y;
    let tid = ly * 16u + lx; // Linear thread ID (0..255)

    // 4x4 Output accumulators in GPU private registers
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    // Number of K-dimension tiles (tile width = 16)
    let num_tiles = (dims.K + 15u) / 16u;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // ---------------------------------------------------------------------
        // 1. Cooperative Load: Tile A [64, 16] (256 threads load 1024 floats)
        // ---------------------------------------------------------------------
        let load_a_row = tid / 4u;             // 0..63
        let load_a_col_vec = (tid % 4u) * 4u;  // 0, 4, 8, 12
        let g_a_row = wg_id.y * 64u + load_a_row;
        let g_a_col = t * 16u + load_a_col_vec;

        if (g_a_row < dims.M) {
            for (var c: u32 = 0u; c < 4u; c = c + 1u) {
                if (g_a_col + c < dims.K) {
                    tile_a[load_a_row][load_a_col_vec + c] = A[g_a_row * dims.K + g_a_col + c];
                } else {
                    tile_a[load_a_row][load_a_col_vec + c] = 0.0;
                }
            }
        } else {
            for (var c: u32 = 0u; c < 4u; c = c + 1u) {
                tile_a[load_a_row][load_a_col_vec + c] = 0.0;
            }
        }

        // ---------------------------------------------------------------------
        // 2. Cooperative Load: Tile B [16, 64] (256 threads load 1024 floats)
        // ---------------------------------------------------------------------
        let load_b_row = tid / 16u;            // 0..15
        let load_b_col_vec = (tid % 16u) * 4u; // 0, 4, ..., 60
        let g_b_row = t * 16u + load_b_row;
        let g_b_col = wg_id.x * 64u + load_b_col_vec;

        if (g_b_row < dims.K) {
            for (var c: u32 = 0u; c < 4u; c = c + 1u) {
                if (g_b_col + c < dims.N) {
                    tile_b[load_b_row][load_b_col_vec + c] = B[g_b_row * dims.N + g_b_col + c];
                } else {
                    tile_b[load_b_row][load_b_col_vec + c] = 0.0;
                }
            }
        } else {
            for (var c: u32 = 0u; c < 4u; c = c + 1u) {
                tile_b[load_b_row][load_b_col_vec + c] = 0.0;
            }
        }

        workgroupBarrier();

        // ---------------------------------------------------------------------
        // 3. Compute Phase: 4x4 Register-Tiled Outer Product Loop
        // ---------------------------------------------------------------------
        let row_base = ly * 4u;
        let col_base = lx * 4u;

        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let b_vec = vec4<f32>(
                tile_b[k][col_base],
                tile_b[k][col_base + 1u],
                tile_b[k][col_base + 2u],
                tile_b[k][col_base + 3u]
            );

            let a0 = tile_a[row_base][k];
            let a1 = tile_a[row_base + 1u][k];
            let a2 = tile_a[row_base + 2u][k];
            let a3 = tile_a[row_base + 3u][k];

            acc0 = acc0 + a0 * b_vec;
            acc1 = acc1 + a1 * b_vec;
            acc2 = acc2 + a2 * b_vec;
            acc3 = acc3 + a3 * b_vec;
        }

        workgroupBarrier();
    }

    // -------------------------------------------------------------------------
    // 4. Writeback Phase: Store 4x4 Register Tile to Global Memory C
    // -------------------------------------------------------------------------
    let global_out_row = wg_id.y * 64u + ly * 4u;
    let global_out_col = wg_id.x * 64u + lx * 4u;

    // Row 0
    if (global_out_row < dims.M) {
        if (global_out_col < dims.N)      { C[global_out_row * dims.N + global_out_col] = acc0.x; }
        if (global_out_col + 1u < dims.N) { C[global_out_row * dims.N + global_out_col + 1u] = acc0.y; }
        if (global_out_col + 2u < dims.N) { C[global_out_row * dims.N + global_out_col + 2u] = acc0.z; }
        if (global_out_col + 3u < dims.N) { C[global_out_row * dims.N + global_out_col + 3u] = acc0.w; }
    }

    // Row 1
    if (global_out_row + 1u < dims.M) {
        let r1 = global_out_row + 1u;
        if (global_out_col < dims.N)      { C[r1 * dims.N + global_out_col] = acc1.x; }
        if (global_out_col + 1u < dims.N) { C[r1 * dims.N + global_out_col + 1u] = acc1.y; }
        if (global_out_col + 2u < dims.N) { C[r1 * dims.N + global_out_col + 2u] = acc1.z; }
        if (global_out_col + 3u < dims.N) { C[r1 * dims.N + global_out_col + 3u] = acc1.w; }
    }

    // Row 2
    if (global_out_row + 2u < dims.M) {
        let r2 = global_out_row + 2u;
        if (global_out_col < dims.N)      { C[r2 * dims.N + global_out_col] = acc2.x; }
        if (global_out_col + 1u < dims.N) { C[r2 * dims.N + global_out_col + 1u] = acc2.y; }
        if (global_out_col + 2u < dims.N) { C[r2 * dims.N + global_out_col + 2u] = acc2.z; }
        if (global_out_col + 3u < dims.N) { C[r2 * dims.N + global_out_col + 3u] = acc2.w; }
    }

    // Row 3
    if (global_out_row + 3u < dims.M) {
        let r3 = global_out_row + 3u;
        if (global_out_col < dims.N)      { C[r3 * dims.N + global_out_col] = acc3.x; }
        if (global_out_col + 1u < dims.N) { C[r3 * dims.N + global_out_col + 1u] = acc3.y; }
        if (global_out_col + 2u < dims.N) { C[r3 * dims.N + global_out_col + 2u] = acc3.z; }
        if (global_out_col + 3u < dims.N) { C[r3 * dims.N + global_out_col + 3u] = acc3.w; }
    }
}
