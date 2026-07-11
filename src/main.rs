use clap::Parser;
use rand::seq::SliceRandom;
use rand::Rng;
use rocr::data::idx::{Idx1, Idx3};
use rocr::math::loss::{CrossEntropyLoss, Softmax};
use rocr::math::optimizer::SGD;
use rocr::math::tensor::Tensor;
use rocr::nets::convolutional_layer::{Activation as ConvActivation, ConvolutionalLayer};
use rocr::nets::dense_layer::{Activation as DenseActivation, DenseLayer};
use rocr::nets::flatten_layer::FlattenLayer;
use rocr::nets::layers::Layer;
use rocr::nets::pooling_layer::PoolingLayer;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const MNIST_MEAN: f32 = 0.1307;
const MNIST_STD: f32 = 0.3081;
const MODEL_CONV1_FILTERS: usize = 16;
const MODEL_CONV2_FILTERS: usize = 32;
const MODEL_FLATTEN_SIZE: usize = 5 * 5 * MODEL_CONV2_FILTERS;
const MODEL_DENSE1_SIZE: usize = 256;
const MODEL_DENSE2_SIZE: usize = 128;

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
    #[arg(long)]
    momentum: Option<f32>,
    #[arg(long, default_value = "1.0")]
    lr_decay: f32,
    #[arg(long, default_value = "0.1")]
    val_split: f32,
    #[arg(long, default_value = "0")]
    lr_plateau_patience: usize,
    #[arg(long, default_value = "0.5")]
    lr_plateau_factor: f32,
    #[arg(long, default_value = "0.00001")]
    min_learning_rate: f32,
    #[arg(long, default_value = "0.0")]
    dropout: f32,
    #[arg(long, default_value = "0.0")]
    weight_decay: f32,
    /// Max random pixel shift applied during training (0 = disabled, 2 is a reasonable value).
    #[arg(long, default_value = "0")]
    aug_shift: i32,
    #[arg(long)]
    num_train_images: Option<usize>,
    #[arg(long, default_value = "experiments")]
    output_dir: PathBuf,
}

struct LeNetOptimizers {
    conv1: SGD<f32>,
    conv2: SGD<f32>,
    dense1: SGD<f32>,
    dense2: SGD<f32>,
    dense3: SGD<f32>,
}

impl LeNetOptimizers {
    fn new(learning_rate: f32, momentum: Option<f32>, weight_decay: f32) -> Self {
        fn make_sgd(learning_rate: f32, momentum: Option<f32>, weight_decay: f32) -> SGD<f32> {
            let mut sgd = if let Some(m) = momentum {
                SGD::with_momentum(learning_rate, m)
            } else {
                SGD::new(learning_rate)
            };
            sgd.weight_decay = weight_decay;
            sgd
        }

        LeNetOptimizers {
            conv1: make_sgd(learning_rate, momentum, weight_decay),
            conv2: make_sgd(learning_rate, momentum, weight_decay),
            dense1: make_sgd(learning_rate, momentum, weight_decay),
            dense2: make_sgd(learning_rate, momentum, weight_decay),
            dense3: make_sgd(learning_rate, momentum, weight_decay),
        }
    }

    fn set_learning_rate(&mut self, learning_rate: f32) {
        self.conv1.learning_rate = learning_rate;
        self.conv2.learning_rate = learning_rate;
        self.dense1.learning_rate = learning_rate;
        self.dense2.learning_rate = learning_rate;
        self.dense3.learning_rate = learning_rate;
    }
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
            conv1: ConvolutionalLayer::with_activation(
                MODEL_CONV1_FILTERS,
                vec![3, 3, 1],
                1,
                1,
                ConvActivation::Relu,
            ),
            pool1: PoolingLayer::new(vec![2, 2], 2),
            conv2: ConvolutionalLayer::with_activation(
                MODEL_CONV2_FILTERS,
                vec![5, 5, MODEL_CONV1_FILTERS],
                1,
                0,
                ConvActivation::Relu,
            ),
            pool2: PoolingLayer::new(vec![2, 2], 2),
            flatten: FlattenLayer::new(),
            dense1: DenseLayer::with_activation(
                MODEL_FLATTEN_SIZE,
                MODEL_DENSE1_SIZE,
                DenseActivation::Relu,
            ),
            dense2: DenseLayer::with_activation(
                MODEL_DENSE1_SIZE,
                MODEL_DENSE2_SIZE,
                DenseActivation::Relu,
            ),
            dense3: DenseLayer::new(MODEL_DENSE2_SIZE, 10),
        }
    }

    fn new_with_dropout(dropout_rate: f32) -> Self {
        let rate = if dropout_rate > 0.0 {
            Some(dropout_rate)
        } else {
            None
        };
        let mut lenet = Self::new();
        if let Some(r) = rate {
            lenet.dense1 = DenseLayer::with_dropout(
                MODEL_FLATTEN_SIZE,
                MODEL_DENSE1_SIZE,
                DenseActivation::Relu,
                r,
            );
            lenet.dense2 = DenseLayer::with_dropout(
                MODEL_DENSE1_SIZE,
                MODEL_DENSE2_SIZE,
                DenseActivation::Relu,
                r,
            );
        }
        lenet
    }

    fn set_training(&mut self, training: bool) {
        self.dense1.set_training(training);
        self.dense2.set_training(training);
        self.dense3.set_training(training);
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

    fn update_weights(&mut self, optimizers: &mut LeNetOptimizers) {
        optimizers
            .dense3
            .update_weights(&mut self.dense3.weights, &self.dense3.d_weights);
        optimizers
            .dense3
            .update_bias(&mut self.dense3.bias, &self.dense3.d_bias);

        optimizers
            .dense2
            .update_weights(&mut self.dense2.weights, &self.dense2.d_weights);
        optimizers
            .dense2
            .update_bias(&mut self.dense2.bias, &self.dense2.d_bias);

        optimizers
            .dense1
            .update_weights(&mut self.dense1.weights, &self.dense1.d_weights);
        optimizers
            .dense1
            .update_bias(&mut self.dense1.bias, &self.dense1.d_bias);

        optimizers
            .conv2
            .update_weights(&mut self.conv2.weights, &self.conv2.d_weights);
        optimizers
            .conv2
            .update_bias(&mut self.conv2.bias, &self.conv2.d_bias);

        optimizers
            .conv1
            .update_weights(&mut self.conv1.weights, &self.conv1.d_weights);
        optimizers
            .conv1
            .update_bias(&mut self.conv1.bias, &self.conv1.d_bias);
    }

    fn save_weights(&self, dir: &PathBuf) -> std::io::Result<()> {
        save_tensor(dir, "conv1_weights", &self.conv1.weights)?;
        save_tensor(dir, "conv1_bias", &self.conv1.bias)?;

        save_tensor(dir, "conv2_weights", &self.conv2.weights)?;
        save_tensor(dir, "conv2_bias", &self.conv2.bias)?;

        save_tensor(dir, "dense1_weights", &self.dense1.weights)?;
        save_tensor(dir, "dense1_bias", &self.dense1.bias)?;

        save_tensor(dir, "dense2_weights", &self.dense2.weights)?;
        save_tensor(dir, "dense2_bias", &self.dense2.bias)?;

        save_tensor(dir, "dense3_weights", &self.dense3.weights)?;
        save_tensor(dir, "dense3_bias", &self.dense3.bias)?;

        Ok(())
    }
}

fn save_tensor(dir: &PathBuf, name: &str, tensor: &Tensor<f32>) -> std::io::Result<()> {
    let file_path = dir.join(format!("{}.txt", name));
    let mut file = File::create(file_path)?;
    writeln!(file, "shape={:?}", tensor.shape())?;
    for value in tensor.get_data() {
        writeln!(file, "{:.8}", value)?;
    }
    Ok(())
}

fn normalized_image_tensor(all_images: &[u8], data_idx: usize, image_size: usize) -> Tensor<f32> {
    let start = data_idx * image_size;
    let end = start + image_size;
    let img_data = &all_images[start..end];
    let img_normalized: Vec<f32> = img_data
        .iter()
        .map(|&x| {
            let x01 = (x as f32) / 255.0;
            (x01 - MNIST_MEAN) / MNIST_STD
        })
        .collect();
    Tensor::new(img_normalized, vec![28, 28, 1])
}

/// Returns a normalized 28×28×1 tensor with a random pixel-level integer shift applied.
/// Pixels shifted outside the image boundary are filled with the background value
/// (MNIST background ≈ 0, which after normalization equals `(0.0 - MNIST_MEAN) / MNIST_STD`).
fn augmented_image_tensor(
    all_images: &[u8],
    data_idx: usize,
    image_size: usize,
    max_shift: i32,
) -> Tensor<f32> {
    const W: i32 = 28;
    const H: i32 = 28;
    let bg = (0.0_f32 - MNIST_MEAN) / MNIST_STD;

    let mut rng = rand::thread_rng();
    let shift_x: i32 = rng.gen_range(-max_shift..=max_shift);
    let shift_y: i32 = rng.gen_range(-max_shift..=max_shift);

    let start = data_idx * image_size;
    let src = &all_images[start..start + image_size];

    let mut out = vec![bg; (W * H) as usize];
    for row in 0..H {
        for col in 0..W {
            let src_row = row - shift_y;
            let src_col = col - shift_x;
            if src_row >= 0 && src_row < H && src_col >= 0 && src_col < W {
                let src_idx = (src_row * W + src_col) as usize;
                let x01 = (src[src_idx] as f32) / 255.0;
                out[(row * W + col) as usize] = (x01 - MNIST_MEAN) / MNIST_STD;
            }
        }
    }
    Tensor::new(out, vec![28, 28, 1])
}

fn one_hot_target(label_val: usize, classes: usize) -> Tensor<f32> {
    let mut target = Tensor::new(vec![0.0f32; classes], vec![classes]);
    target[&[label_val]] = 1.0;
    target
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
    println!("Base learning rate: {}", args.learning_rate);
    println!("Momentum: {:?}", args.momentum);
    println!("LR decay: {}", args.lr_decay);
    println!("Dropout: {}", args.dropout);
    println!("Weight decay: {}", args.weight_decay);
    println!("Aug shift: {}", args.aug_shift);
    println!("Validation split: {}", args.val_split);
    println!("LR plateau patience: {}", args.lr_plateau_patience);
    println!("LR plateau factor: {}", args.lr_plateau_factor);
    println!("Min learning rate: {}", args.min_learning_rate);
    println!("Num train images: {:?}", args.num_train_images);
    println!("Output dir: {:?}", args.output_dir);
    println!();

    if args.lr_decay <= 0.0 {
        panic!("--lr-decay must be > 0");
    }

    if !(0.0..1.0).contains(&args.val_split) {
        panic!("--val-split must be in [0.0, 1.0)");
    }

    if args.lr_plateau_factor <= 0.0 || args.lr_plateau_factor >= 1.0 {
        panic!("--lr-plateau-factor must be in (0.0, 1.0)");
    }

    if args.dropout < 0.0 || args.dropout >= 1.0 {
        panic!("--dropout must be in [0.0, 1.0)");
    }

    if args.weight_decay < 0.0 {
        panic!("--weight-decay must be >= 0");
    }

    if args.min_learning_rate <= 0.0 {
        panic!("--min-learning-rate must be > 0");
    }

    if args.aug_shift < 0 {
        panic!("--aug-shift must be >= 0");
    }

    if let Some(m) = args.momentum {
        if !(0.0..1.0).contains(&m) {
            panic!("--momentum must be in [0.0, 1.0)");
        }
    }

    if let Some(n) = args.num_train_images {
        if n == 0 {
            panic!("--num-train-images must be > 0");
        }
    }

    // Load MNIST data headers and full dataset into memory.
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

    let max_samples = (image_header.num_images as usize).min(label_header.num_labels as usize);
    let num_samples = args
        .num_train_images
        .unwrap_or(max_samples)
        .min(max_samples);
    let image_size = (image_header.shape.0 * image_header.shape.1) as usize;

    let all_images = image_header.read_next_images(&mut images_reader, num_samples)?;
    let mut all_labels = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        all_labels.push(Idx1::read_next_label(&mut labels_reader)? as usize);
    }

    let val_samples = if num_samples > 1 {
        (((num_samples as f32) * args.val_split).round() as usize)
            .max(1)
            .min(num_samples - 1)
    } else {
        0
    };
    let train_samples = num_samples - val_samples;

    println!("Using {} samples total", num_samples);
    println!("Train/Val split: {}/{}\n", train_samples, val_samples);

    // Initialize model
    let mut model = LeNet::new_with_dropout(args.dropout);
    let mut optimizers = LeNetOptimizers::new(args.learning_rate, args.momentum, args.weight_decay);

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let run_dir = args.output_dir.join(format!("run_{}", timestamp));
    let weights_dir = run_dir.join("weights");
    fs::create_dir_all(&weights_dir)?;

    let metrics_path = run_dir.join("metrics.csv");
    let mut metrics_file = File::create(&metrics_path)?;
    writeln!(
        metrics_file,
        "epoch,learning_rate,train_loss,train_accuracy_percent,val_loss,val_accuracy_percent,train_samples,val_samples"
    )?;

    // Training loop
    let mut scheduled_lr = args.learning_rate;
    let mut best_val_loss = f32::INFINITY;
    let mut plateau_epochs = 0usize;

    for epoch in 0..args.num_epochs {
        let current_lr = scheduled_lr;
        optimizers.set_learning_rate(current_lr);

        let mut train_total_loss = 0.0f32;
        let mut train_total_samples = 0;
        let mut train_correct = 0;

        let mut indices: Vec<usize> = (0..num_samples).collect();
        indices.shuffle(&mut rand::thread_rng());
        // Enable dropout during training, disable for validation.
        model.set_training(true);

        let (train_indices, val_indices) = indices.split_at(train_samples);

        let num_batches = train_samples.div_ceil(args.batch_size);
        for batch_idx in 0..num_batches {
            let mut batch_loss = 0.0f32;
            let mut batch_correct = 0;
            let mut batch_sample_count = 0;

            for sample_idx in 0..args.batch_size {
                let actual_idx = batch_idx * args.batch_size + sample_idx;
                if actual_idx >= train_samples {
                    break;
                }

                let data_idx = train_indices[actual_idx];

                let mut image = if args.aug_shift > 0 {
                    augmented_image_tensor(&all_images, data_idx, image_size, args.aug_shift)
                } else {
                    normalized_image_tensor(&all_images, data_idx, image_size)
                };

                let label_val = all_labels[data_idx];

                // Create one-hot target
                let target = one_hot_target(label_val, 10);

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
                model.update_weights(&mut optimizers);
                batch_sample_count += 1;
            }

            train_total_loss += batch_loss;
            train_correct += batch_correct;
            train_total_samples += batch_sample_count;

            if batch_sample_count > 0 && (batch_idx + 1) % 5 == 0 {
                let avg_batch_loss = batch_loss / (batch_sample_count as f32);
                println!("  Batch {}: Loss = {:.6}", batch_idx + 1, avg_batch_loss);
            }
        }

        if train_total_samples == 0 {
            println!("Epoch {} complete: no samples processed\n", epoch + 1);
            continue;
        }

        let train_avg_loss = train_total_loss / (train_total_samples as f32);
        let train_accuracy = train_correct as f32 / (train_total_samples as f32);

        // Disable dropout for validation pass.
        model.set_training(false);

        // Validation pass (no gradient updates)
        let mut val_total_loss = 0.0f32;
        let mut val_correct = 0usize;
        let mut val_count = 0usize;
        for &data_idx in val_indices {
            let mut image = normalized_image_tensor(&all_images, data_idx, image_size);
            let label_val = all_labels[data_idx];
            let target = one_hot_target(label_val, 10);

            let logits = model.forward(&mut image);
            let predictions = Softmax::forward(&logits);

            val_total_loss += CrossEntropyLoss::forward(&predictions, &target);
            if argmax_1d(&predictions) == label_val {
                val_correct += 1;
            }
            val_count += 1;
        }

        let val_avg_loss = if val_count > 0 {
            val_total_loss / (val_count as f32)
        } else {
            0.0
        };
        let val_accuracy = if val_count > 0 {
            (val_correct as f32) / (val_count as f32)
        } else {
            0.0
        };

        println!(
            "Epoch {} complete (lr={:.6}): Train Loss = {:.6}, Train Acc = {:.2}%, Val Loss = {:.6}, Val Acc = {:.2}%\n",
            epoch + 1,
            current_lr,
            train_avg_loss,
            train_accuracy * 100.0,
            val_avg_loss,
            val_accuracy * 100.0
        );

        let mut metrics_file = OpenOptions::new().append(true).open(&metrics_path)?;
        writeln!(
            metrics_file,
            "{},{:.8},{:.8},{:.4},{:.8},{:.4},{},{}",
            epoch + 1,
            current_lr,
            train_avg_loss,
            train_accuracy * 100.0,
            val_avg_loss,
            val_accuracy * 100.0,
            train_total_samples,
            val_count
        )?;

        if val_count > 0 {
            if val_avg_loss < best_val_loss {
                best_val_loss = val_avg_loss;
                plateau_epochs = 0;
            } else {
                plateau_epochs += 1;
                if args.lr_plateau_patience > 0 && plateau_epochs >= args.lr_plateau_patience {
                    scheduled_lr =
                        (scheduled_lr * args.lr_plateau_factor).max(args.min_learning_rate);
                    plateau_epochs = 0;
                    println!("  LR reduced on plateau to {:.6}", scheduled_lr);
                }
            }
        }

        // Apply base multiplicative decay each epoch (can combine with plateau reductions).
        scheduled_lr = (scheduled_lr * args.lr_decay).max(args.min_learning_rate);
    }

    model.save_weights(&weights_dir)?;

    let mut summary_file = File::create(run_dir.join("summary.txt"))?;
    writeln!(summary_file, "run_dir={}", run_dir.display())?;
    writeln!(summary_file, "train_images={}", args.train_images.display())?;
    writeln!(summary_file, "train_labels={}", args.train_labels.display())?;
    writeln!(summary_file, "num_epochs={}", args.num_epochs)?;
    writeln!(summary_file, "batch_size={}", args.batch_size)?;
    writeln!(summary_file, "learning_rate={}", args.learning_rate)?;
    writeln!(summary_file, "momentum={:?}", args.momentum)?;
    writeln!(summary_file, "lr_decay={}", args.lr_decay)?;
    writeln!(summary_file, "dropout={}", args.dropout)?;
    writeln!(summary_file, "weight_decay={}", args.weight_decay)?;
    writeln!(summary_file, "aug_shift={}", args.aug_shift)?;
    writeln!(summary_file, "val_split={}", args.val_split)?;
    writeln!(
        summary_file,
        "lr_plateau_patience={}",
        args.lr_plateau_patience
    )?;
    writeln!(summary_file, "lr_plateau_factor={}", args.lr_plateau_factor)?;
    writeln!(summary_file, "min_learning_rate={}", args.min_learning_rate)?;
    writeln!(summary_file, "num_train_images={:?}", args.num_train_images)?;
    writeln!(summary_file, "used_samples={}", num_samples)?;
    writeln!(summary_file, "train_samples={}", train_samples)?;
    writeln!(summary_file, "val_samples={}", val_samples)?;
    writeln!(summary_file, "weights_dir={}", weights_dir.display())?;
    writeln!(summary_file, "metrics_file={}", metrics_path.display())?;

    println!("Training completed!");
    println!("Saved results to: {}", run_dir.display());
    Ok(())
}
