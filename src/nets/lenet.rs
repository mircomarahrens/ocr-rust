use crate::math::tensor::Tensor;
use crate::math::optimizer::SGD;
use crate::nets::convolutional_layer::{Activation as ConvActivation, ConvolutionalLayer};
use crate::nets::dense_layer::{Activation as DenseActivation, DenseLayer};
use crate::nets::flatten_layer::FlattenLayer;
use crate::nets::pooling_layer::PoolingLayer;
use crate::nets::layers::Layer;
use std::path::PathBuf;
use std::fs::File;
use std::io::Write;

pub const MNIST_MEAN: f32 = 0.1307;
pub const MNIST_STD: f32 = 0.3081;
pub const MODEL_CONV1_FILTERS: usize = 16;
pub const MODEL_CONV2_FILTERS: usize = 32;
pub const MODEL_FLATTEN_SIZE: usize = 5 * 5 * MODEL_CONV2_FILTERS;
pub const MODEL_DENSE1_SIZE: usize = 256;
pub const MODEL_DENSE2_SIZE: usize = 128;

#[derive(Clone)]
pub struct LeNet {
    pub conv1: ConvolutionalLayer<f32>,
    pub pool1: PoolingLayer,
    pub conv2: ConvolutionalLayer<f32>,
    pub pool2: PoolingLayer,
    pub flatten: FlattenLayer,
    pub dense1: DenseLayer<f32>,
    pub dense2: DenseLayer<f32>,
    pub dense3: DenseLayer<f32>,
}

impl LeNet {
    pub fn new() -> Self {
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

    pub fn new_with_dropout(dropout_rate: f32) -> Self {
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

    pub fn set_training(&mut self, training: bool) {
        self.dense1.set_training(training);
        self.dense2.set_training(training);
        self.dense3.set_training(training);
    }

    pub fn forward(&mut self, input: &mut Tensor<f32>) -> Tensor<f32> {
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

    pub fn backward(&mut self, mut d_out: Tensor<f32>) -> Tensor<f32> {
        let mut d_d3 = self.dense3.backward(&mut d_out);
        let mut d_d2 = self.dense2.backward(&mut d_d3);
        let mut d_d1 = self.dense1.backward(&mut d_d2);
        let mut d_f = self.flatten.backward(&mut d_d1);
        let mut d_p2 = self.pool2.backward(&mut d_f);
        let mut d_c2 = self.conv2.backward(&mut d_p2);
        let mut d_p1 = self.pool1.backward(&mut d_c2);
        self.conv1.backward(&mut d_p1)
    }

    pub fn update_weights(&mut self, optimizers: &mut LeNetOptimizers) {
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

    pub fn save_weights(&self, dir: &PathBuf) -> std::io::Result<()> {
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

    pub fn load_weights(&mut self, dir: &std::path::Path) -> std::io::Result<()> {
        self.conv1.weights = load_tensor(dir, "conv1_weights")?;
        self.conv1.bias = load_tensor(dir, "conv1_bias")?;
        self.conv2.weights = load_tensor(dir, "conv2_weights")?;
        self.conv2.bias = load_tensor(dir, "conv2_bias")?;
        self.dense1.weights = load_tensor(dir, "dense1_weights")?;
        self.dense1.bias = load_tensor(dir, "dense1_bias")?;
        self.dense2.weights = load_tensor(dir, "dense2_weights")?;
        self.dense2.bias = load_tensor(dir, "dense2_bias")?;
        self.dense3.weights = load_tensor(dir, "dense3_weights")?;
        self.dense3.bias = load_tensor(dir, "dense3_bias")?;
        Ok(())
    }

    pub fn zero_gradients(&mut self) {
        self.conv1.d_weights.get_data_mut().fill(0.0);
        self.conv1.d_bias.get_data_mut().fill(0.0);
        self.conv2.d_weights.get_data_mut().fill(0.0);
        self.conv2.d_bias.get_data_mut().fill(0.0);
        self.dense1.d_weights.get_data_mut().fill(0.0);
        self.dense1.d_bias.get_data_mut().fill(0.0);
        self.dense2.d_weights.get_data_mut().fill(0.0);
        self.dense2.d_bias.get_data_mut().fill(0.0);
        self.dense3.d_weights.get_data_mut().fill(0.0);
        self.dense3.d_bias.get_data_mut().fill(0.0);
    }

    pub fn accumulate_gradients(&mut self, local_models: &[LeNet]) {
        self.zero_gradients();
        if local_models.is_empty() {
            return;
        }

        let scale = 1.0 / (local_models.len() as f32);

        // Sum gradients
        for local in local_models {
            accumulate_tensor(&mut self.conv1.d_weights, &local.conv1.d_weights);
            accumulate_tensor(&mut self.conv1.d_bias, &local.conv1.d_bias);
            accumulate_tensor(&mut self.conv2.d_weights, &local.conv2.d_weights);
            accumulate_tensor(&mut self.conv2.d_bias, &local.conv2.d_bias);
            accumulate_tensor(&mut self.dense1.d_weights, &local.dense1.d_weights);
            accumulate_tensor(&mut self.dense1.d_bias, &local.dense1.d_bias);
            accumulate_tensor(&mut self.dense2.d_weights, &local.dense2.d_weights);
            accumulate_tensor(&mut self.dense2.d_bias, &local.dense2.d_bias);
            accumulate_tensor(&mut self.dense3.d_weights, &local.dense3.d_weights);
            accumulate_tensor(&mut self.dense3.d_bias, &local.dense3.d_bias);
        }

        // Scale by batch size to get the average gradient
        scale_tensor(&mut self.conv1.d_weights, scale);
        scale_tensor(&mut self.conv1.d_bias, scale);
        scale_tensor(&mut self.conv2.d_weights, scale);
        scale_tensor(&mut self.conv2.d_bias, scale);
        scale_tensor(&mut self.dense1.d_weights, scale);
        scale_tensor(&mut self.dense1.d_bias, scale);
        scale_tensor(&mut self.dense2.d_weights, scale);
        scale_tensor(&mut self.dense2.d_bias, scale);
        scale_tensor(&mut self.dense3.d_weights, scale);
        scale_tensor(&mut self.dense3.d_bias, scale);
    }
}

pub struct LeNetOptimizers {
    pub conv1: SGD<f32>,
    pub conv2: SGD<f32>,
    pub dense1: SGD<f32>,
    pub dense2: SGD<f32>,
    pub dense3: SGD<f32>,
}

impl LeNetOptimizers {
    pub fn new(learning_rate: f32, momentum: Option<f32>, weight_decay: f32) -> Self {
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

    pub fn set_learning_rate(&mut self, learning_rate: f32) {
        self.conv1.learning_rate = learning_rate;
        self.conv2.learning_rate = learning_rate;
        self.dense1.learning_rate = learning_rate;
        self.dense2.learning_rate = learning_rate;
        self.dense3.learning_rate = learning_rate;
    }
}

fn accumulate_tensor(dest: &mut Tensor<f32>, src: &Tensor<f32>) {
    let dest_data = dest.get_data_mut();
    let src_data = src.get_data();
    for (d, s) in dest_data.iter_mut().zip(src_data.iter()) {
        *d += *s;
    }
}

fn scale_tensor(tensor: &mut Tensor<f32>, scale: f32) {
    let data = tensor.get_data_mut();
    for x in data.iter_mut() {
        *x *= scale;
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

fn load_tensor(dir: &std::path::Path, name: &str) -> std::io::Result<Tensor<f32>> {
    use std::io::BufRead;
    let file_path = dir.join(format!("{}.txt", name));
    let file = File::open(file_path)?;
    let reader = std::io::BufReader::new(file);
    let mut lines = reader.lines();

    let first_line = lines.next().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Empty weight file"))??;
    let shape_str = first_line.strip_prefix("shape=[").and_then(|s| s.strip_suffix("]")).ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid shape format"))?;
    let shape: Vec<usize> = shape_str.split(", ").filter_map(|s| s.parse().ok()).collect();

    let mut data = Vec::new();
    for line in lines {
        let line = line?;
        let val: f32 = line.trim().parse().map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        data.push(val);
    }
    Ok(Tensor::new(data, shape))
}
