use crate::nets::layers::Layer;

use num::traits::{Float, FromPrimitive};
use rand::distributions::{Distribution, Standard};
use std::iter::Sum;

use crate::math::linalg::{matmul, matmul_transpose_a, matmul_transpose_b, mul_vals};
use crate::math::tensor::Tensor;
use crate::nets::im2col::{calc_col_shape, calc_w_row, calc_x_col, col2im};

// Add an activation enum
#[derive(Copy, Clone)]
pub enum Activation {
    Sigmoid,
    Relu,
    Tanh,
}

// Convolutional layer for a neural network.
pub struct ConvolutionalLayer<T>
where
    T: Float + FromPrimitive,
{
    num_filters: usize,              // Number of filters, K
    spatial_extent: Vec<usize>,      // Spatial extent of the filters, F [fw, fh, d]
    stride: usize,                   // Stride for the filters, S
    zero_padding: usize,             // Zero padding for the input volume
    activation: Option<Activation>,  // Optional activation
    pub weights: Tensor<T>,          // shape [num_filters, filter_size]
    pub bias: Tensor<T>,             // shape [num_filters]
    pub d_weights: Tensor<T>,        // gradient accumulator [num_filters, filter_size]
    pub d_bias: Tensor<T>,           // gradient accumulator [num_filters]
    x_col_cache: Option<Tensor<T>>,  // im2col output cached during forward
    output_cache: Option<Tensor<T>>, // post-activation output cached during forward
    input_shape_cache: Vec<usize>,   // original (unpadded) input shape
}

impl<T> ConvolutionalLayer<T>
where
    T: Float + Sum + FromPrimitive,
    Standard: Distribution<T>,
{
    // Constructor for a convolutional layer.
    pub fn new(
        num_filters: usize,
        spatial_extent: Vec<usize>,
        stride: usize,
        zero_padding: usize,
    ) -> Self {
        let filter_size = mul_vals(&spatial_extent);
        let weights = calc_w_row(num_filters, filter_size);
        let bias = Tensor::new(vec![T::zero(); num_filters], vec![num_filters]);
        let d_weights = Tensor::new(
            vec![T::zero(); num_filters * filter_size],
            vec![num_filters, filter_size],
        );
        let d_bias = Tensor::new(vec![T::zero(); num_filters], vec![num_filters]);
        ConvolutionalLayer {
            num_filters,
            spatial_extent,
            stride,
            zero_padding,
            activation: None,
            weights,
            bias,
            d_weights,
            d_bias,
            x_col_cache: None,
            output_cache: None,
            input_shape_cache: Vec::new(),
        }
    }

    // Constructor with activation
    pub fn with_activation(
        num_filters: usize,
        spatial_extent: Vec<usize>,
        stride: usize,
        zero_padding: usize,
        activation: Activation,
    ) -> Self {
        let filter_size = mul_vals(&spatial_extent);
        let weights = calc_w_row(num_filters, filter_size);
        let bias = Tensor::new(vec![T::zero(); num_filters], vec![num_filters]);
        let d_weights = Tensor::new(
            vec![T::zero(); num_filters * filter_size],
            vec![num_filters, filter_size],
        );
        let d_bias = Tensor::new(vec![T::zero(); num_filters], vec![num_filters]);
        ConvolutionalLayer {
            num_filters,
            spatial_extent,
            stride,
            zero_padding,
            activation: Some(activation),
            weights,
            bias,
            d_weights,
            d_bias,
            x_col_cache: None,
            output_cache: None,
            input_shape_cache: Vec::new(),
        }
    }
}

// Helper to apply activation in-place
fn apply_activation_inplace<T: Float>(v: &mut Tensor<T>, act: Activation) {
    let shape = v.shape().to_vec();
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    for i in 0..h {
        for j in 0..w {
            for k in 0..c {
                let x = v[&[i, j, k]];
                let y = match act {
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
                };
                v[&[i, j, k]] = y;
            }
        }
    }
}

impl<T> Layer<T> for ConvolutionalLayer<T>
where
    T: Float + Sum + FromPrimitive,
    Standard: Distribution<T>,
{
    fn forward(&mut self, input_volume: &mut Tensor<T>) -> Tensor<T> {
        let input_shape = input_volume.shape();
        assert!(
            input_shape.len() == 3 && self.spatial_extent.len() == 3,
            "Convolution expects rank-3 input and filter shapes [W, H, D]."
        );
        self.input_shape_cache.clone_from(&input_shape);

        // pad only spatial dimensions
        input_volume.pad(self.zero_padding, T::zero());

        let filter_size = mul_vals(&self.spatial_extent);
        let col_shape = calc_col_shape(input_volume, &self.spatial_extent, self.stride);
        let out_w = col_shape[0];
        let out_h = col_shape[1];

        let x_col = calc_x_col(
            input_volume,
            &col_shape,
            filter_size,
            &self.spatial_extent,
            self.stride,
        );

        // Y = weights @ x_col  shape [num_filters, out_w * out_h]
        let y = matmul(&self.weights, &x_col);

        // Build output [out_w, out_h, num_filters] with correct layout.
        // output[w, h, f] = y[f, w * out_h + h] + bias[f]
        let mut output_data = vec![T::zero(); out_w * out_h * self.num_filters];
        for f in 0..self.num_filters {
            let b = self.bias[&[f]];
            for w in 0..out_w {
                for h in 0..out_h {
                    let out_flat = (w * out_h + h) * self.num_filters + f;
                    output_data[out_flat] = y[&[f, w * out_h + h]] + b;
                }
            }
        }
        let mut output_volume = Tensor::new(output_data, vec![out_w, out_h, self.num_filters]);

        // apply activation if configured
        if let Some(act) = self.activation {
            apply_activation_inplace(&mut output_volume, act);
        }

        // cache for backward
        self.x_col_cache = Some(x_col);
        self.output_cache = Some(output_volume.clone());

        output_volume
    }

    fn backward(&mut self, d_out: &mut Tensor<T>) -> Tensor<T> {
        let x_col = self
            .x_col_cache
            .as_ref()
            .expect("backward called before forward")
            .clone();
        let out_cache = self
            .output_cache
            .as_ref()
            .expect("backward called before forward")
            .clone();
        let original_shape = self.input_shape_cache.clone();

        let out_w = d_out.shape()[0];
        let out_h = d_out.shape()[1];
        let out_positions = out_w * out_h;

        // Step 1: activation backward — modify d_out in-place using cached post-activation output
        if let Some(act) = self.activation {
            for w in 0..out_w {
                for h in 0..out_h {
                    for f in 0..self.num_filters {
                        let cached = out_cache[&[w, h, f]];
                        let grad = d_out[&[w, h, f]];
                        d_out[&[w, h, f]] = match act {
                            Activation::Relu => {
                                if cached > T::zero() {
                                    grad
                                } else {
                                    T::zero()
                                }
                            }
                            Activation::Sigmoid => grad * cached * (T::one() - cached),
                            Activation::Tanh => grad * (T::one() - cached * cached),
                        };
                    }
                }
            }
        }

        // Step 2: convert d_out [out_w, out_h, num_filters] -> d_y [num_filters, out_positions]
        // mirrors the layout built in forward: output[w, h, f] <-> y[f, w * out_h + h]
        let mut d_y_data = vec![T::zero(); self.num_filters * out_positions];
        for f in 0..self.num_filters {
            for w in 0..out_w {
                for h in 0..out_h {
                    d_y_data[f * out_positions + w * out_h + h] = d_out[&[w, h, f]];
                }
            }
        }
        let d_y = Tensor::new(d_y_data, vec![self.num_filters, out_positions]);

        // Step 3: dW = d_y @ x_col^T  [num_filters, filter_size]
        self.d_weights = matmul_transpose_b(&d_y, &x_col);

        // Step 4: db[f] = sum(d_y[f, :])
        for f in 0..self.num_filters {
            let mut s = T::zero();
            for p in 0..out_positions {
                s = s + d_y[&[f, p]];
            }
            self.d_bias[&[f]] = s;
        }

        // Step 5: d_x_col = W^T @ d_y  [filter_size, out_positions]
        let d_x_col = matmul_transpose_a(&self.weights, &d_y);

        // Step 6: col2im -> d_x_pad [W+2P, H+2P, D]
        let padded_shape = vec![
            original_shape[0] + 2 * self.zero_padding,
            original_shape[1] + 2 * self.zero_padding,
            original_shape[2],
        ];
        let d_x_pad = col2im(
            &d_x_col,
            &padded_shape,
            &self.spatial_extent,
            self.stride,
            out_w,
            out_h,
        );

        // Step 7: unpad -> [W, H, D]
        let p = self.zero_padding;
        d_x_pad.slice(&[
            p..p + original_shape[0],
            p..p + original_shape[1],
            0..original_shape[2],
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward() {
        // Test the forward pass of the convolutional layer
        let num_filters = 96;
        let spatial_extent = vec![11, 11, 3];
        let stride = 4;
        let zero_padding = 0;

        let mut conv =
            ConvolutionalLayer::new(num_filters, spatial_extent.clone(), stride, zero_padding);

        let w_1 = 227;
        let h_1 = 227;
        let d_1 = 3;

        let shape = vec![w_1, h_1, d_1];
        let data_size = mul_vals(&shape);
        let data = (1..data_size + 1).map(|v| v as f32).collect::<Vec<f32>>();
        let mut input_volume = Tensor::new(data, shape.clone());

        let output_volume = conv.forward(&mut input_volume);

        let w_2 = (w_1 - spatial_extent[0] + 2 * zero_padding) / stride + 1;
        let h_2 = (h_1 - spatial_extent[1] + 2 * zero_padding) / stride + 1;
        let d_2 = num_filters;

        assert_eq!(output_volume.shape(), vec![w_2, h_2, d_2]);
    }

    #[test]
    fn test_forward_with_activation() {
        // Test the forward pass of the convolutional layer with activation
        let num_filters = 96;
        let spatial_extent = vec![11, 11, 3];
        let stride = 4;
        let zero_padding = 0;

        let mut conv = ConvolutionalLayer::with_activation(
            num_filters,
            spatial_extent,
            stride,
            zero_padding,
            Activation::Relu,
        );

        let shape = vec![227, 227, 3];
        let data_size = mul_vals(&shape);
        let data = (1..data_size + 1).map(|v| v as f32).collect::<Vec<f32>>();
        let mut input_volume = Tensor::new(data, shape.clone());

        let output_volume = conv.forward(&mut input_volume);

        assert_eq!(output_volume.shape(), vec![55, 55, 96]);

        // Check that all values are non-negative due to ReLU
        for val in output_volume.get_data() {
            assert!(*val >= 0.0);
        }
    }

    #[test]
    fn test_forward_deterministic() {
        // Two forward passes with the same weights and input must produce identical output.
        let num_filters = 4;
        let spatial_extent = vec![3, 3, 1];
        let stride = 1;
        let zero_padding = 1;

        let mut conv: ConvolutionalLayer<f32> =
            ConvolutionalLayer::new(num_filters, spatial_extent, stride, zero_padding);

        let shape = vec![8, 8, 1];
        let data_size = mul_vals(&shape);
        let data: Vec<f32> = (0..data_size).map(|v| v as f32).collect();

        let out1 = conv.forward(&mut Tensor::new(data.clone(), shape.clone()));
        let out2 = conv.forward(&mut Tensor::new(data.clone(), shape.clone()));

        assert_eq!(out1.get_data(), out2.get_data());
    }

    #[test]
    fn test_forward_grayscale_shape_small() {
        let num_filters = 6;
        let spatial_extent = vec![3, 3, 1];
        let stride = 1;
        let zero_padding = 1;

        let mut conv: ConvolutionalLayer<f32> =
            ConvolutionalLayer::new(num_filters, spatial_extent.clone(), stride, zero_padding);

        let shape = vec![28, 28, 1];
        let data_size = mul_vals(&shape);
        let data = (0..data_size).map(|v| v as f32).collect::<Vec<f32>>();
        let mut input_volume = Tensor::new(data, shape);

        let output_volume = conv.forward(&mut input_volume);

        assert_eq!(output_volume.shape(), vec![28, 28, num_filters]);
    }

    #[test]
    fn test_backward() {
        // Shape test: d_input must match the original input shape.
        let num_filters = 2;
        let spatial_extent = vec![3, 3, 1];
        let stride = 1;
        let zero_padding = 0;

        let mut conv: ConvolutionalLayer<f32> =
            ConvolutionalLayer::new(num_filters, spatial_extent, stride, zero_padding);

        let input_shape = vec![5, 5, 1];
        let data_size = mul_vals(&input_shape);
        let data: Vec<f32> = (0..data_size).map(|v| v as f32).collect();
        let mut input_volume = Tensor::new(data, input_shape.clone());

        let output = conv.forward(&mut input_volume);

        let d_out_shape = output.shape();
        let d_out_size: usize = d_out_shape.iter().product();
        let mut d_out = Tensor::new(vec![1.0f32; d_out_size], d_out_shape);

        let d_input = conv.backward(&mut d_out);

        assert_eq!(d_input.shape(), input_shape);
        // d_weights and d_bias must be populated
        assert_eq!(conv.d_weights.shape(), conv.weights.shape());
        assert_eq!(conv.d_bias.shape(), conv.bias.shape());
    }
}
