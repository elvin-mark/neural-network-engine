use neural_network_engine::prelude::*;

#[test]
fn test_tensor_pool_recycle_and_hit_rate() {
    TensorPool::clear_local();

    // 1. Initial allocation (cold miss)
    let shape = vec![64, 64];
    {
        let t1 = RawTensor::zeros(&shape);
        assert_eq!(t1.shape(), &[64, 64]);
        // t1 is dropped here, its underlying Vec<f32> is recycled into TensorPool!
    }

    let stats_after_first = TensorPool::local_stats();
    assert!(stats_after_first.cached_bytes > 0);
    assert_eq!(stats_after_first.free_buffers, 1);

    // 2. Second allocation with matching size (cache hit)
    {
        let t2 = RawTensor::zeros(&shape);
        assert_eq!(t2.shape(), &[64, 64]);
    }

    let stats_after_second = TensorPool::local_stats();
    assert_eq!(stats_after_second.hits, 1);
    assert!(stats_after_second.hit_rate() > 0.0);
}

#[test]
fn test_tensor_pool_clear() {
    TensorPool::clear_local();

    {
        let _t = RawTensor::zeros(&[128, 128]);
    }
    assert!(TensorPool::local_stats().free_buffers > 0);

    TensorPool::clear_local();
    assert_eq!(TensorPool::local_stats().free_buffers, 0);
    assert_eq!(TensorPool::local_stats().cached_bytes, 0);
}

#[test]
fn test_tensor_pool_in_training_loop() {
    TensorPool::clear_local();

    let linear = Linear::new(32, 64);
    let x = Tensor::randn(&[8, 32], 0.0, 1.0, true);

    // Run 5 training steps
    for _ in 0..5 {
        let out = linear.forward(&x).unwrap();
        let loss = out.sum_all();
        loss.backward();
    }

    let stats = TensorPool::local_stats();
    assert!(
        stats.hits > 0,
        "Expected cache hits during training loop, got {}",
        stats
    );
}
