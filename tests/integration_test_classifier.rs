use rocr::math::tensor::Tensor;
use rocr::nets::dense_layer::{Activation, DenseLayer};
use rocr::nets::flatten_layer::FlattenLayer;
use rocr::nets::layers::Layer;

#[test]
fn test_classifier_head_pipeline() {
    // Simulate post-pooling output: [5, 5, 16] after second pooling layer
    let post_pool_shape = vec![5, 5, 16];
    let post_pool_size: usize = post_pool_shape.iter().product();
    let post_pool_data: Vec<f32> = (0..post_pool_size).map(|v| (v as f32) * 0.1).collect();
    let mut post_pool = Tensor::new(post_pool_data, post_pool_shape);

    // Step 1: Flatten [5, 5, 16] → [400]
    let mut flatten = FlattenLayer::new();
    let flat_output = flatten.forward(&mut post_pool);
    assert_eq!(
        flat_output.shape(),
        vec![400],
        "Flatten output should be [400]"
    );

    // Step 2: Dense 400 → 120 with sigmoid
    let mut dense1: DenseLayer<f32> = DenseLayer::with_activation(400, 120, Activation::Sigmoid);
    let mut flat_for_dense1 = flat_output.clone();
    let dense1_output = dense1.forward(&mut flat_for_dense1);
    assert_eq!(
        dense1_output.shape(),
        vec![120],
        "Dense1 output should be [120]"
    );
    // Check activation: all values should be in [0, 1] due to sigmoid
    for val in dense1_output.get_data() {
        assert!(
            *val >= 0.0 && *val <= 1.0,
            "Sigmoid output should be in [0, 1], got {}",
            val
        );
    }

    // Step 3: Dense 120 → 84 with sigmoid
    let mut dense2: DenseLayer<f32> = DenseLayer::with_activation(120, 84, Activation::Sigmoid);
    let mut dense1_for_dense2 = dense1_output.clone();
    let dense2_output = dense2.forward(&mut dense1_for_dense2);
    assert_eq!(
        dense2_output.shape(),
        vec![84],
        "Dense2 output should be [84]"
    );
    // Check activation
    for val in dense2_output.get_data() {
        assert!(
            *val >= 0.0 && *val <= 1.0,
            "Sigmoid output should be in [0, 1], got {}",
            val
        );
    }

    // Step 4: Dense 84 → 10 (no activation for logits)
    let mut dense3: DenseLayer<f32> = DenseLayer::new(84, 10);
    let mut dense2_for_dense3 = dense2_output.clone();
    let logits = dense3.forward(&mut dense2_for_dense3);
    assert_eq!(logits.shape(), vec![10], "Output logits should be [10]");

    println!("✓ Classifier head pipeline works: [5,5,16] → [400] → [120] → [84] → [10]");
}
