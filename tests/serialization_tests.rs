use neural_network_engine::prelude::*;
use std::collections::HashMap;

#[test]
fn test_safetensors_roundtrip() {
    let mut map = HashMap::new();
    let t1 = RawTensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let t2 = RawTensor::randn(&[3, 4, 5], 0.0, 1.0);

    map.insert("weight1".to_string(), t1.clone());
    map.insert("weight2".to_string(), t2.clone());

    let file_path = "test_roundtrip.safetensors";
    save_safetensors(&map, file_path).unwrap();

    let loaded = load_safetensors(file_path).unwrap();
    let _ = std::fs::remove_file(file_path);

    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded.get("weight1").unwrap().as_slice(), t1.as_slice());
    assert_eq!(loaded.get("weight2").unwrap().shape(), t2.shape());
}

#[test]
fn test_checkpoint_roundtrip() {
    let mut cp = Checkpoint::new(5, 1200);
    let t = RawTensor::from_slice(&[42.0, 84.0], &[2]);
    cp.insert("param", &t);

    let json_path = "test_cp.json";
    cp.save_json(json_path).unwrap();
    let loaded_cp = Checkpoint::load_json(json_path).unwrap();
    let _ = std::fs::remove_file(json_path);

    assert_eq!(loaded_cp.epoch, 5);
    assert_eq!(loaded_cp.step, 1200);
    assert_eq!(loaded_cp.get("param").unwrap().as_slice(), &[42.0, 84.0]);

    let bin_path = "test_cp.bin";
    cp.save_bincode(bin_path).unwrap();
    let loaded_bin = Checkpoint::load_bincode(bin_path).unwrap();
    let _ = std::fs::remove_file(bin_path);

    assert_eq!(loaded_bin.epoch, 5);
    assert_eq!(loaded_bin.step, 1200);
    assert_eq!(loaded_bin.get("param").unwrap().as_slice(), &[42.0, 84.0]);
}
