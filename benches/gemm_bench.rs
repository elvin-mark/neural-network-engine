use criterion::{black_box, criterion_group, criterion_main, Criterion};
use neural_network_engine::prelude::*;

fn bench_gemm_512(c: &mut Criterion) {
    let a = RawTensor::randn(&[512, 512], 0.0, 1.0);
    let b = RawTensor::randn(&[512, 512], 0.0, 1.0);

    c.bench_function("gemm_512x512", |bencher| {
        bencher.iter(|| {
            let out = neural_network_engine::tensor::matmul::matmul(black_box(&a), black_box(&b))
                .unwrap();
            black_box(out);
        });
    });
}

fn bench_mlp_forward_backward(c: &mut Criterion) {
    let x = Tensor::randn(&[64, 128], 0.0, 1.0, false);
    let l1 = Linear::new(128, 256);
    let l2 = Linear::new(256, 10);

    c.bench_function("mlp_64x128_fwd_bwd", |bencher| {
        bencher.iter(|| {
            let h = l1.forward(black_box(&x)).unwrap().relu().unwrap();
            let out = l2.forward(&h).unwrap();
            let loss = out.sum_all();
            loss.backward();
            l1.zero_grad();
            l2.zero_grad();
        });
    });
}

criterion_group!(benches, bench_gemm_512, bench_mlp_forward_backward);
criterion_main!(benches);
