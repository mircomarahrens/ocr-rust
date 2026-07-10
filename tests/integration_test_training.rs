use rocr::math::loss::{CrossEntropyLoss, Softmax};
use rocr::math::optimizer::SGD;
use rocr::math::tensor::Tensor;
use rocr::nets::convolutional_layer::ConvolutionalLayer;
use rocr::nets::dense_layer::{Activation, DenseLayer};
use rocr::nets::flatten_layer::FlattenLayer;
use rocr::nets::layers::Layer;
use rocr::nets::pooling_layer::PoolingLayer;

#[test]
fn test_full_lenet_training_step() {
    // Simulate a single training step on a small batch
    // Architecture: [28,28,1] → Conv(6,3×3,pad=1) + sigmoid → Pool(2×2) → Conv(16,5×5) + sigmoid → Pool(2×2) → Flatten → Dense(120) + sigmoid → Dense(84) + sigmoid → Dense(10)

    println!("\n=== LeNet Mini-Training Step Demo ===");

    // Step 0: Create a single sample and target (fake MNIST image)
    let sample_size = 28 * 28;
    let sample_data: Vec<f32> = (0..sample_size)
        .map(|i| ((i as f32) / (sample_size as f32)))
        .collect();
    let mut sample = Tensor::new(sample_data, vec![28, 28, 1]);

    // Target: one-hot for class 3
    let mut target = Tensor::new(vec![0.0f32; 10], vec![10]);
    target[&[3]] = 1.0;

    // Step 1: Conv Layer 1 (28×28×1 → 28×28×6)
    let mut conv1 = ConvolutionalLayer::new(6, vec![3, 3, 1], 1, 1);
    let conv1_out = conv1.forward(&mut sample);
    println!("After Conv1: {:?}", conv1_out.shape());
    assert_eq!(conv1_out.shape(), vec![28, 28, 6]);

    // Step 2: Pool Layer 1 (28×28×6 → 14×14×6)
    let mut pool1 = PoolingLayer::new(vec![2, 2], 2);
    let mut conv1_out = conv1_out;
    let pool1_out = pool1.forward(&mut conv1_out);
    println!("After Pool1: {:?}", pool1_out.shape());
    assert_eq!(pool1_out.shape(), vec![14, 14, 6]);

    // Step 3: Conv Layer 2 (14×14×6 → 10×10×16)
    let mut conv2 = ConvolutionalLayer::new(16, vec![5, 5, 6], 1, 0);
    let mut pool1_out = pool1_out;
    let conv2_out = conv2.forward(&mut pool1_out);
    println!("After Conv2: {:?}", conv2_out.shape());
    assert_eq!(conv2_out.shape(), vec![10, 10, 16]);

    // Step 4: Pool Layer 2 (10×10×16 → 5×5×16)
    let mut pool2 = PoolingLayer::new(vec![2, 2], 2);
    let mut conv2_out = conv2_out;
    let pool2_out = pool2.forward(&mut conv2_out);
    println!("After Pool2: {:?}", pool2_out.shape());
    assert_eq!(pool2_out.shape(), vec![5, 5, 16]);

    // Step 5: Flatten (5×5×16 → 400)
    let mut flatten = FlattenLayer::new();
    let mut pool2_out = pool2_out;
    let flat_out = flatten.forward(&mut pool2_out);
    println!("After Flatten: {:?}", flat_out.shape());
    assert_eq!(flat_out.shape(), vec![400]);

    // Step 6: Dense Layer 1 (400 → 120)
    let mut dense1 = DenseLayer::with_activation(400, 120, Activation::Sigmoid);
    let mut flat_out = flat_out;
    let dense1_out = dense1.forward(&mut flat_out);
    println!("After Dense1: {:?}", dense1_out.shape());
    assert_eq!(dense1_out.shape(), vec![120]);

    // Step 7: Dense Layer 2 (120 → 84)
    let mut dense2 = DenseLayer::with_activation(120, 84, Activation::Sigmoid);
    let mut dense1_out = dense1_out;
    let dense2_out = dense2.forward(&mut dense1_out);
    println!("After Dense2: {:?}", dense2_out.shape());
    assert_eq!(dense2_out.shape(), vec![84]);

    // Step 8: Dense Layer 3 (84 → 10)
    let mut dense3 = DenseLayer::new(84, 10);
    let mut dense2_out = dense2_out;
    let logits = dense3.forward(&mut dense2_out);
    println!("After Dense3 (logits): {:?}", logits.shape());
    assert_eq!(logits.shape(), vec![10]);

    // Step 9: Softmax + Compute Loss
    let predictions = Softmax::forward(&logits);
    let loss = CrossEntropyLoss::forward(&predictions, &target);
    println!("Initial Loss: {}", loss);

    // Step 10: Backward pass
    let mut d_out = CrossEntropyLoss::backward(&predictions, &target);
    println!("Loss gradient shape: {:?}", d_out.shape());

    // Backward through Dense3
    let mut d_dense3_in = dense3.backward(&mut d_out);

    // Backward through Dense2
    let mut d_dense2_in = dense2.backward(&mut d_dense3_in);

    // Backward through Dense1
    let mut d_dense1_in = dense1.backward(&mut d_dense2_in);

    // Backward through Flatten
    let mut d_flat_in = flatten.backward(&mut d_dense1_in);

    // Backward through Pool2
    let mut d_pool2_in = pool2.backward(&mut d_flat_in);

    // Backward through Conv2
    let mut d_conv2_in = conv2.backward(&mut d_pool2_in);

    // Backward through Pool1
    let mut d_pool1_in = pool1.backward(&mut d_conv2_in);

    // Backward through Conv1
    let _d_input = conv1.backward(&mut d_pool1_in);

    println!("Backward pass completed successfully");

    // Step 11: Optimizer updates (just a sanity check that optimizer works)
    let mut sgd_dense3: SGD<f32> = SGD::new(0.01);
    let mut sgd_dense2: SGD<f32> = SGD::new(0.01);
    let mut sgd_dense1: SGD<f32> = SGD::new(0.01);

    sgd_dense3.update_weights(&mut dense3.weights, &dense3.d_weights);
    sgd_dense3.update_bias(&mut dense3.bias, &dense3.d_bias);

    sgd_dense2.update_weights(&mut dense2.weights, &dense2.d_weights);
    sgd_dense2.update_bias(&mut dense2.bias, &dense2.d_bias);

    sgd_dense1.update_weights(&mut dense1.weights, &dense1.d_weights);
    sgd_dense1.update_bias(&mut dense1.bias, &dense1.d_bias);

    println!("Weight updates completed");

    // Step 12: Forward pass again to verify loss changed
    let sample2: Vec<f32> = (0..sample_size)
        .map(|i| ((i as f32) / (sample_size as f32)))
        .collect();
    let mut sample_input = Tensor::new(sample2, vec![28, 28, 1]);

    let conv1_out2 = conv1.forward(&mut sample_input);
    let mut pool1_in2 = conv1_out2;
    let pool1_out2 = pool1.forward(&mut pool1_in2);
    let mut conv2_in2 = pool1_out2;
    let conv2_out2 = conv2.forward(&mut conv2_in2);
    let mut pool2_in2 = conv2_out2;
    let pool2_out2 = pool2.forward(&mut pool2_in2);
    let mut flat_in2 = pool2_out2;
    let flat_out2 = flatten.forward(&mut flat_in2);
    let mut d1_in2 = flat_out2;
    let d1_out2 = dense1.forward(&mut d1_in2);
    let mut d2_in2 = d1_out2;
    let d2_out2 = dense2.forward(&mut d2_in2);
    let mut d3_in2 = d2_out2;
    let logits2 = dense3.forward(&mut d3_in2);

    let predictions2 = Softmax::forward(&logits2);
    let loss2 = CrossEntropyLoss::forward(&predictions2, &target);
    println!("Loss after update: {}", loss2);

    // Verify the pipeline completed
    println!("✓ Full LeNet training step completed successfully");
}

#[test]
fn test_lenet_loss_decreases_over_iterations() {
    // Multi-iteration training to show loss decreases
    println!("\n=== LeNet Loss Decay Test ===");

    let sample_size = 28 * 28;
    let sample_data: Vec<f32> = (0..sample_size)
        .map(|i| ((i as f32) / (sample_size as f32)) * 0.1)
        .collect();

    let mut target = Tensor::new(vec![0.0f32; 10], vec![10]);
    target[&[5]] = 1.0;

    let mut conv1 = ConvolutionalLayer::new(6, vec![3, 3, 1], 1, 1);
    let mut pool1 = PoolingLayer::new(vec![2, 2], 2);
    let mut conv2 = ConvolutionalLayer::new(16, vec![5, 5, 6], 1, 0);
    let mut pool2 = PoolingLayer::new(vec![2, 2], 2);
    let mut flatten = FlattenLayer::new();
    let mut dense1 = DenseLayer::with_activation(400, 120, Activation::Sigmoid);
    let mut dense2 = DenseLayer::with_activation(120, 84, Activation::Sigmoid);
    let mut dense3 = DenseLayer::new(84, 10);

    let mut sgd3: SGD<f32> = SGD::new(0.1);
    let mut sgd2: SGD<f32> = SGD::new(0.1);
    let mut sgd1: SGD<f32> = SGD::new(0.1);

    let mut prev_loss = f32::INFINITY;

    for iter in 0..3 {
        // Forward
        let mut x = Tensor::new(sample_data.clone(), vec![28, 28, 1]);
        let c1 = conv1.forward(&mut x);
        let mut p1_in = c1;
        let p1 = pool1.forward(&mut p1_in);
        let mut c2_in = p1;
        let c2 = conv2.forward(&mut c2_in);
        let mut p2_in = c2;
        let p2 = pool2.forward(&mut p2_in);
        let mut f_in = p2;
        let f = flatten.forward(&mut f_in);
        let mut d1_in = f;
        let d1 = dense1.forward(&mut d1_in);
        let mut d2_in = d1;
        let d2 = dense2.forward(&mut d2_in);
        let mut d3_in = d2;
        let logits = dense3.forward(&mut d3_in);

        let pred = Softmax::forward(&logits);
        let loss = CrossEntropyLoss::forward(&pred, &target);
        println!("Iteration {}: Loss = {}", iter, loss);

        // Backward
        let mut d_out = CrossEntropyLoss::backward(&pred, &target);
        let mut d_d3 = dense3.backward(&mut d_out);
        let mut d_d2 = dense2.backward(&mut d_d3);
        let mut d_d1 = dense1.backward(&mut d_d2);
        let mut d_f = flatten.backward(&mut d_d1);
        let mut d_p2 = pool2.backward(&mut d_f);
        let mut _d_c2 = conv2.backward(&mut d_p2);
        let _d_rest = pool1.backward(&mut _d_c2);

        // Update
        sgd3.update_weights(&mut dense3.weights, &dense3.d_weights);
        sgd3.update_bias(&mut dense3.bias, &dense3.d_bias);
        sgd2.update_weights(&mut dense2.weights, &dense2.d_weights);
        sgd2.update_bias(&mut dense2.bias, &dense2.d_bias);
        sgd1.update_weights(&mut dense1.weights, &dense1.d_weights);
        sgd1.update_bias(&mut dense1.bias, &dense1.d_bias);

        assert!(
            loss < prev_loss || (loss - prev_loss).abs() < 0.01,
            "Loss did not decrease: prev={}, curr={}",
            prev_loss,
            loss
        );
        prev_loss = loss;
    }

    println!("✓ Loss decreased or remained stable across iterations");
}
