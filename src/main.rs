use clap::Parser;
use rocr::data::idx::{Idx1, Idx3};
use rocr::math::loss::{CrossEntropyLoss, Softmax};
use rocr::math::optimizer::SGD;
use rocr::math::tensor::Tensor;
use rocr::nets::convolutional_layer::ConvolutionalLayer;
use rocr::nets::dense_layer::{Activation, DenseLayer};
use rocr::nets::flatten_layer::FlattenLayer;
use rocr::nets::layers::Layer;
use rocr::nets::pooling_layer::PoolingLayer;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    train_images: PathBuf,
    #[arg(long)]
    train_labels: PathBuf,
    #[arg(long, default_value = "5")]
    num_epochs: usize,
    #[arg(long, default_value = "32")]
    batch_size: usize,
    #[arg(long, default_value = "0.01")]
    learning_rate: f32,
}

struct LeNet {
    conv1: ConvolutionalLayer<f32>,
    pool1: PoolingLayer,
    conv2: ConvolutionalLayer<f32>,
    pool2: PoolingLayer,
    flatten: FlattenLayer,
    dense1: DenseLayer<f32>,
    dense2: DenseLayer<f32>,
    dense3: DenseLayer<f32>,
}

fn argmax_1d(tensor: &Tensor<f32>) -> usize {
    let mut max_idx = 0usize;
    let mut max_val = tensor[&[0]];
    for i in 1..10 {
        if tensor[&[i]] > max_val {
            max_val = tensor[&[i]];
            max_idx = i;
        }
    }
    max_idx
}

impl LeNet {
    fn new() -> Self {
        LeNet {
            conv1: ConvolutionalLayer::new(6, vec![3, 3, 1], 1, 1),
            pool1: PoolingLayer::new(vec![2, 2], 2),
            conv2: ConvolutionalLayer::new(16, vec![5, 5, 6], 1, 0),
            pool2: PoolingLayer::new(vec![2, 2], 2),
            flatten: FlattenLayer::new(),
            dense1: DenseLayer::with_activation(400, 120, Activation::Sigmoid),
            dense2: DenseLayer::with_activation(120, 84, Activation::Sigmoid),
            dense3: DenseLayer::new(84, 10),
        }
    }

    fn forward(&mut self, input: &mut Tensor<f32>) -> Tensor<f32> {
        let c1 = self.conv1.forward(input);
        let mut p1_in = c1;
        let p1 = self.pool1.forward(&mut p1_in);
        let mut c2_in = p1;
        let c2 = self.conv2.forward(&mut c2_in);
        let mut p2_in = c2;
        let p2 = self.pool2.forward(&mut p2_in);
        let mut f_in = p2;
        let f = self.flatten.forward(&mut f_in);
        let mut d1_in = f;
        let d1 = self.dense1.forward(&mut d1_in);
        let mut d2_in = d1;
        let d2 = self.dense2.forward(&mut d2_in);
        let mut d3_in = d2;
        self.dense3.forward(&mut d3_in)
    }

    fn backward(&mut self, mut d_out: Tensor<f32>) -> Tensor<f32> {
        let mut d_d3 = self.dense3.backward(&mut d_out);
        let mut d_d2 = self.dense2.backward(&mut d_d3);
        let mut d_d1 = self.dense1.backward(&mut d_d2);
        let mut d_f = self.flatten.backward(&mut d_d1);
        let mut d_p2 = self.pool2.backward(&mut d_f);
        let mut d_c2 = self.conv2.backward(&mut d_p2);
        let mut d_p1 = self.pool1.backward(&mut d_c2);
        self.conv1.backward(&mut d_p1)
    }

    fn update_weights(&mut self, learning_rate: f32) {
        let mut sgd = SGD::new(learning_rate);

        sgd.update_weights(&mut self.dense3.weights, &self.dense3.d_weights);
        sgd.update_bias(&mut self.dense3.bias, &self.dense3.d_bias);

        sgd.update_weights(&mut self.dense2.weights, &self.dense2.d_weights);
        sgd.update_bias(&mut self.dense2.bias, &self.dense2.d_bias);

        sgd.update_weights(&mut self.dense1.weights, &self.dense1.d_weights);
        sgd.update_bias(&mut self.dense1.bias, &self.dense1.d_bias);

        sgd.update_weights(&mut self.conv2.weights, &self.conv2.d_weights);
        sgd.update_bias(&mut self.conv2.bias, &self.conv2.d_bias);

        sgd.update_weights(&mut self.conv1.weights, &self.conv1.d_weights);
        sgd.update_bias(&mut self.conv1.bias, &self.conv1.d_bias);
    }
}

fn main() -> std::io::Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("LeNet Training on MNIST");
    println!("=======================");
    println!("Train images: {:?}", args.train_images);
    println!("Train labels: {:?}", args.train_labels);
    println!("Epochs: {}", args.num_epochs);
    println!("Batch size: {}", args.batch_size);
    println!("Learning rate: {}", args.learning_rate);
    println!();

    // Load MNIST data headers
    let images_file = File::open(&args.train_images)?;
    let mut images_reader = BufReader::new(images_file);

    let labels_file = File::open(&args.train_labels)?;
    let mut labels_reader = BufReader::new(labels_file);

    let image_header = Idx3::read_header(&mut images_reader)?;
    let label_header = Idx1::read_header(&mut labels_reader)?;

    println!(
        "Loaded {} training images ({}x{})",
        image_header.num_images, image_header.shape.0, image_header.shape.1
    );
    println!("Loaded {} training labels", label_header.num_labels);

    let num_samples = (image_header.num_images as usize).min(500); // Limit to 500 for demo
    println!("Using {} samples for training\n", num_samples);

    // Initialize model
    let mut model = LeNet::new();

    // Training loop
    for epoch in 0..args.num_epochs {
        let mut total_loss = 0.0f32;
        let mut total_samples = 0;
        let mut correct = 0;

        // Reset readers for new epoch
        let images_file = File::open(&args.train_images)?;
        let mut images_reader = BufReader::new(images_file);
        let labels_file = File::open(&args.train_labels)?;
        let mut labels_reader = BufReader::new(labels_file);

        Idx3::read_header(&mut images_reader)?;
        Idx1::read_header(&mut labels_reader)?;

        let num_batches = num_samples.div_ceil(args.batch_size);
        for batch_idx in 0..num_batches {
            let mut batch_loss = 0.0f32;
            let mut batch_correct = 0;
            let mut batch_sample_count = 0;

            for sample_idx in 0..args.batch_size {
                let actual_idx = batch_idx * args.batch_size + sample_idx;
                if actual_idx >= num_samples {
                    break;
                }

                // Read image
                let img_data = image_header.read_next_image(&mut images_reader)?;
                let img_normalized: Vec<f32> =
                    img_data.iter().map(|&x| (x as f32) / 255.0).collect();
                let mut image = Tensor::new(img_normalized, vec![28, 28, 1]);

                // Read label from file
                let label_val = Idx1::read_next_label(&mut labels_reader)? as usize;

                // Create one-hot target
                let mut target = Tensor::new(vec![0.0f32; 10], vec![10]);
                target[&[label_val]] = 1.0;

                // Forward pass
                let logits = model.forward(&mut image);
                let predictions = Softmax::forward(&logits);

                // Compute loss
                let loss = CrossEntropyLoss::forward(&predictions, &target);
                batch_loss += loss;

                // Get prediction accuracy
                let max_idx = argmax_1d(&predictions);
                if max_idx == label_val {
                    batch_correct += 1;
                }

                // Backward pass
                let d_out = CrossEntropyLoss::backward(&predictions, &target);
                let _d_input = model.backward(d_out);

                // Update weights
                model.update_weights(args.learning_rate);
                batch_sample_count += 1;
            }

            total_loss += batch_loss;
            correct += batch_correct;
            total_samples += batch_sample_count;

            if batch_sample_count > 0 && (batch_idx + 1) % 5 == 0 {
                let avg_batch_loss = batch_loss / (batch_sample_count as f32);
                println!("  Batch {}: Loss = {:.6}", batch_idx + 1, avg_batch_loss);
            }
        }

        if total_samples == 0 {
            println!("Epoch {} complete: no samples processed\n", epoch + 1);
            continue;
        }

        let avg_loss = total_loss / (total_samples as f32);
        let accuracy = correct as f32 / (total_samples as f32);
        println!(
            "Epoch {} complete: Avg Loss = {:.6}, Accuracy = {:.2}%\n",
            epoch + 1,
            avg_loss,
            accuracy * 100.0
        );
    }

    println!("Training completed!");
    Ok(())
}
