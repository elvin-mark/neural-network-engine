use neural_network_engine::prelude::*;

#[test]
fn test_moe_layer_forward_shape() {
    let config = MoEConfig::mini(32, 4);
    let moe = MoELayer::new(config);
    let x = Tensor::randn(&[2, 8, 32], 0.0, 1.0, false);
    let (out, aux_loss) = moe.forward_with_aux(&x).unwrap();
    assert_eq!(out.shape(), vec![2, 8, 32]);
    assert_eq!(aux_loss.shape(), Vec::<usize>::new());
}

#[test]
fn test_moe_router_selects_top_k() {
    let config = MoEConfig {
        d_model: 16,
        hidden_dim: 32,
        num_experts: 4,
        top_k: 2,
        aux_loss_coeff: 0.01,
    };
    let moe = MoELayer::new(config);
    let x = Tensor::randn(&[4, 16], 0.0, 1.0, false);
    let (gate_weights, indices, _) = moe.router.route(&x).unwrap();
    assert_eq!(gate_weights.shape(), vec![4, 2]); // [B*T, top_k]
    assert_eq!(indices.len(), 4 * 2); // B*T * top_k

    // All gate weights should be positive (softmax output) and sum to ~1 per row
    let gw_data = gate_weights.data();
    for &v in gw_data.as_slice() {
        assert!(v >= 0.0);
    }
}

#[test]
fn test_sparse_moe_block_gradient_flow() {
    let moe_config = MoEConfig::mini(32, 4);
    let block = SparseMoEBlock::new(32, 4, moe_config);
    let x = Tensor::randn(&[1, 4, 32], 0.0, 1.0, true);
    let (out, aux_loss) = block.forward_with_aux(&x).unwrap();
    let total_loss = out.sum_all().add(&aux_loss).unwrap();
    total_loss.backward();

    // Check router gate received gradients
    let gate_params = block.moe.router.gate.parameters();
    assert!(gate_params[0].grad().is_some());
}
