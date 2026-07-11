use clap::Parser;
use rand::seq::SliceRandom;
use rand::Rng;
use rocr::data::idx::{Idx1, Idx3};
use rocr::math::loss::{CrossEntropyLoss, Softmax};
use rocr::math::tensor::Tensor;
use rocr::nets::lenet::{
    LeNet, LeNetOptimizers, MNIST_MEAN, MNIST_STD,
};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

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
            let batch_start = batch_idx * args.batch_size;
            let batch_end = (batch_start + args.batch_size).min(train_samples);
            let current_batch_size = batch_end - batch_start;

            if current_batch_size == 0 {
                continue;
            }

            use rayon::prelude::*;
            // Process the batch in parallel
            let batch_results: Vec<(f32, usize, LeNet)> = (batch_start..batch_end)
                .into_par_iter()
                .map(|actual_idx| {
                    let data_idx = train_indices[actual_idx];
                    let mut image = if args.aug_shift > 0 {
                        augmented_image_tensor(&all_images, data_idx, image_size, args.aug_shift)
                    } else {
                        normalized_image_tensor(&all_images, data_idx, image_size)
                    };
                    let label_val = all_labels[data_idx];
                    let target = one_hot_target(label_val, 10);

                    // We clone the model to have thread-local weights & forward/backward state
                    let mut local_model = model.clone();

                    let logits = local_model.forward(&mut image);
                    let predictions = Softmax::forward(&logits);
                    let loss = CrossEntropyLoss::forward(&predictions, &target);

                    let max_idx = argmax_1d(&predictions);
                    let correct = if max_idx == label_val { 1 } else { 0 };

                    let d_out = CrossEntropyLoss::backward(&predictions, &target);
                    let _d_input = local_model.backward(d_out);

                    (loss, correct, local_model)
                })
                .collect();

            // Accumulate metrics and model gradients sequentially
            let mut batch_loss = 0.0f32;
            let mut batch_correct = 0;
            let mut local_models = Vec::with_capacity(current_batch_size);

            for (loss, correct, local_model) in batch_results {
                batch_loss += loss;
                batch_correct += correct;
                local_models.push(local_model);
            }

            // Accumulate all local gradients into the main model and average them
            model.accumulate_gradients(&local_models);

            // Update the main weights once per batch!
            model.update_weights(&mut optimizers);

            train_total_loss += batch_loss;
            train_correct += batch_correct;
            train_total_samples += current_batch_size;

            if current_batch_size > 0 && (batch_idx + 1) % 5 == 0 {
                let avg_batch_loss = batch_loss / (current_batch_size as f32);
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
