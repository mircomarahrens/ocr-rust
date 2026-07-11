use crate::math::linalg::init_random_vec;
use crate::math::tensor::Tensor;
use crate::nets::layers::Layer;
use num::traits::{Float, FromPrimitive};
use rand::distributions::{Distribution, Standard};
use std::iter::Sum;

#[derive(Copy, Clone, Debug)]
pub enum Activation {
    Sigmoid,
    Relu,
    Tanh,
}

fn apply_activation<T: Float>(x: T, act: Activation) -> T {
    match act {
        Activation::Sigmoid => {
            let one = T::one();
            one / (one + (-x).exp())
        }
        Activation::Relu => {
            if x > T::zero() {
                x
            } else {
                T::zero()
            }
        }
        Activation::Tanh => x.tanh(),
    }
}

fn activation_derivative<T: Float>(cached_output: T, act: Activation) -> T {
    match act {
        Activation::Sigmoid => cached_output * (T::one() - cached_output),
        Activation::Relu => {
            if cached_output > T::zero() {
                T::one()
            } else {
                T::zero()
            }
        }
        Activation::Tanh => T::one() - cached_output * cached_output,
    }
}

/// Dense (fully-connected) layer
#[derive(Clone, Debug)]
pub struct DenseLayer<T>
where
    T: Float + FromPrimitive,
{
    pub out_size: usize,
    pub in_size: usize,
    pub weights: Tensor<T>,   // shape [out_size, in_size]
    pub bias: Tensor<T>,      // shape [out_size]
    pub d_weights: Tensor<T>, // shape [out_size, in_size]
    pub d_bias: Tensor<T>,    // shape [out_size]
    activation: Option<Activation>,
    cached_input: Option<Tensor<T>>, // input [in_size] cached for backward
    cached_output: Option<Tensor<T>>, // post-activation output [out_size] cached
    /// Fraction of outputs to randomly zero during training (inverted dropout).
    pub dropout_rate: Option<T>,
    /// True during training; False during inference (no dropout applied).
    pub training: bool,
    dropout_mask: Vec<bool>, // true = keep, false = dropped
}

impl<T> DenseLayer<T>
where
    T: Float + Sum + FromPrimitive,
    Standard: Distribution<T>,
{
    pub fn new(in_size: usize, out_size: usize) -> Self {
        let weights_data: Vec<T> = init_random_vec(out_size * in_size);
        let weights = Tensor::new(weights_data, vec![out_size, in_size]);
        let bias = Tensor::new(vec![T::zero(); out_size], vec![out_size]);
        let d_weights = Tensor::new(vec![T::zero(); out_size * in_size], vec![out_size, in_size]);
        let d_bias = Tensor::new(vec![T::zero(); out_size], vec![out_size]);

        DenseLayer {
            out_size,
            in_size,
            weights,
            bias,
            d_weights,
            d_bias,
            activation: None,
            cached_input: None,
            cached_output: None,
            dropout_rate: None,
            training: true,
            dropout_mask: Vec::new(),
        }
    }

    pub fn with_activation(in_size: usize, out_size: usize, activation: Activation) -> Self {
        let mut layer = Self::new(in_size, out_size);
        layer.activation = Some(activation);
        layer
    }

    pub fn with_dropout(
        in_size: usize,
        out_size: usize,
        activation: Activation,
        dropout_rate: T,
    ) -> Self {
        let mut layer = Self::with_activation(in_size, out_size, activation);
        layer.dropout_rate = Some(dropout_rate);
        layer
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

impl<T> Layer<T> for DenseLayer<T>
where
    T: Float + Sum + FromPrimitive + Send + Sync,
    Standard: Distribution<T>,
{
    fn forward(&mut self, input_volume: &mut Tensor<T>) -> Tensor<T> {
        let input_shape = input_volume.shape();
        assert_eq!(
            input_shape.len(),
            1,
            "Dense layer expects 1D input, got shape {:?}",
            input_shape
        );
        assert_eq!(
            input_shape[0], self.in_size,
            "Dense layer expects input size {}, got {}",
            self.in_size, input_shape[0]
        );

        self.cached_input = Some(input_volume.clone());

        let weights_data = self.weights.get_data();
        let input_data = input_volume.get_data();
        let bias_data = self.bias.get_data();

        use rayon::prelude::*;
        let mut output_data: Vec<T> = (0..self.out_size)
            .into_par_iter()
            .map(|i| {
                let row_offset = i * self.in_size;
                let w_row = &weights_data[row_offset..row_offset + self.in_size];
                let mut sum = bias_data[i];
                for j in 0..self.in_size {
                    sum = sum + w_row[j] * input_data[j];
                }
                if let Some(act) = self.activation {
                    sum = apply_activation(sum, act);
                }
                sum
            })
            .collect();

        // Inverted dropout: applied after activation, only in training mode.
        if self.training {
            if let Some(rate) = self.dropout_rate {
                let scale = T::one() / (T::one() - rate);
                self.dropout_mask = (0..self.out_size)
                    .map(|_| {
                        let sample = T::from_f64(rand::random::<f64>()).unwrap_or(T::zero());
                        sample >= rate
                    })
                    .collect();
                for i in 0..self.out_size {
                    if !self.dropout_mask[i] {
                        output_data[i] = T::zero();
                    } else {
                        output_data[i] = output_data[i] * scale;
                    }
                }
            }
        }

        let output = Tensor::new(output_data, vec![self.out_size]);
        self.cached_output = Some(output.clone());
        output
    }

    fn backward(&mut self, d_out: &mut Tensor<T>) -> Tensor<T> {
        let input = self
            .cached_input
            .as_ref()
            .expect("backward called before forward");
        let output_cache = self
            .cached_output
            .as_ref()
            .expect("backward called before forward");

        let mut d_z_data = d_out.get_data().clone();
        if let Some(act) = self.activation {
            let cached_out_data = output_cache.get_data();
            for i in 0..self.out_size {
                let deriv = activation_derivative(cached_out_data[i], act);
                d_z_data[i] = d_z_data[i] * deriv;
            }
        }

        // Apply dropout mask to d_z so dropped units receive zero gradient.
        if !self.dropout_mask.is_empty() {
            let scale = if let Some(rate) = self.dropout_rate {
                T::one() / (T::one() - rate)
            } else {
                T::one()
            };
            for i in 0..self.out_size {
                if !self.dropout_mask[i] {
                    d_z_data[i] = T::zero();
                } else {
                    d_z_data[i] = d_z_data[i] * scale;
                }
            }
        }

        // Step 2: dW = d_z @ x^T  [out_size, in_size]
        let input_data = input.get_data();
        let d_weights_data = self.d_weights.get_data_mut();

        use rayon::prelude::*;
        d_weights_data
            .par_chunks_exact_mut(self.in_size)
            .enumerate()
            .for_each(|(i, row)| {
                let dz = d_z_data[i];
                for j in 0..self.in_size {
                    row[j] = dz * input_data[j];
                }
            });

        // Step 3: db[i] = d_z[i]
        self.d_bias.get_data_mut().copy_from_slice(&d_z_data);

        // Step 4: d_input = W^T @ d_z  [in_size]
        let weights_data = self.weights.get_data();
        let d_input_data: Vec<T> = (0..self.in_size)
            .into_par_iter()
            .map(|j| {
                let mut sum = T::zero();
                for i in 0..self.out_size {
                    sum = sum + weights_data[i * self.in_size + j] * d_z_data[i];
                }
                sum
            })
            .collect();

        Tensor::new(d_input_data, vec![self.in_size])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_shape() {
        let mut dense: DenseLayer<f32> = DenseLayer::new(10, 5);
        let mut input = Tensor::new((1..=10).map(|x| x as f32).collect::<Vec<_>>(), vec![10]);

        let output = dense.forward(&mut input);

        assert_eq!(output.shape(), vec![5]);
    }

    #[test]
    fn test_forward_with_activation() {
        let mut dense: DenseLayer<f32> = DenseLayer::with_activation(10, 5, Activation::Relu);
        let mut input = Tensor::new((1..=10).map(|x| x as f32).collect::<Vec<_>>(), vec![10]);

        let output = dense.forward(&mut input);

        assert_eq!(output.shape(), vec![5]);
        // All values should be non-negative due to ReLU
        for val in output.get_data() {
            assert!(*val >= 0.0);
        }
    }

    #[test]
    fn test_backward_shape() {
        let mut dense: DenseLayer<f32> = DenseLayer::new(10, 5);
        let mut input = Tensor::new((1..=10).map(|x| x as f32).collect::<Vec<_>>(), vec![10]);

        let _ = dense.forward(&mut input);

        let mut d_out = Tensor::new(vec![1.0f32; 5], vec![5]);
        let d_input = dense.backward(&mut d_out);

        assert_eq!(d_input.shape(), vec![10]);
        assert_eq!(dense.d_weights.shape(), vec![5, 10]);
        assert_eq!(dense.d_bias.shape(), vec![5]);
    }
}
