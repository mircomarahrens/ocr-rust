use crate::nets::layers::Layer;

use num::traits::{Float, FromPrimitive};
use rand::distributions::{Distribution, Standard};
use std::iter::Sum;

use crate::math::linalg::{matmul, mul_vals};
use crate::math::tensor::Tensor;
use crate::nets::im2col::{calc_col_shape, calc_w_row, calc_x_col};

// Add an activation enum
#[derive(Copy, Clone)]
pub enum Activation {
    Sigmoid,
    Relu,
    Tanh,
}

// Convolutional layer for a neural network.
pub struct ConvolutionalLayer {
    num_filters: usize,             // Number of filters, K
    spatial_extent: Vec<usize>,     // Spatial extent of the filters, F
    stride: usize,                  // Stride for the filters, S
    zero_padding: usize,            // Zero padding for the input volume
    activation: Option<Activation>, // Optional activation
}

impl ConvolutionalLayer {
    // Constructor for a convolutional layer.
    pub fn new(
        num_filters: usize,
        spatial_extent: Vec<usize>,
        stride: usize,
        zero_padding: usize,
    ) -> Self {
        ConvolutionalLayer {
            num_filters,
            spatial_extent,
            stride,
            zero_padding,
            activation: None,
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
        ConvolutionalLayer {
            num_filters,
            spatial_extent,
            stride,
            zero_padding,
            activation: Some(activation),
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

impl<T> Layer<T> for ConvolutionalLayer
where
    T: Float + Sum + FromPrimitive,
    Standard: Distribution<T>,
{
    fn forward(&self, input_volume: &mut Tensor<T>) -> Tensor<T> {
        // pad the input volume
        input_volume.pad(self.zero_padding, T::zero());

        // calculate the filter size
        let filter_size = mul_vals(&self.spatial_extent);

        // perform im2col
        let w_row = calc_w_row(self.num_filters, filter_size);

        let col_shape = calc_col_shape(input_volume, &self.spatial_extent, self.stride);

        let x_col = calc_x_col(
            input_volume,
            &col_shape,
            filter_size,
            &self.spatial_extent,
            self.stride,
        );

        // calculate the output volume
        let mut output_volume = matmul(&w_row, &x_col);

        output_volume.reshape(vec![col_shape[0], col_shape[1], self.num_filters]);

        // apply activation if configured
        if let Some(act) = self.activation {
            apply_activation_inplace(&mut output_volume, act);
        }

        output_volume
    }

    fn backward(&self, input_volume: &mut Tensor<T>) -> Tensor<T> {
        // TODO
        Tensor::new(vec![], vec![])
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

        let conv = ConvolutionalLayer::new(num_filters, spatial_extent, stride, zero_padding);

        let shape = vec![227, 227, 3];
        let data_size = mul_vals(&shape);
        let data = (1..data_size + 1).map(|v| v as f32).collect::<Vec<f32>>();
        let mut input_volume = Tensor::new(data, shape.clone());

        let output_volume = conv.forward(&mut input_volume);

        assert_eq!(output_volume.shape(), vec![55, 55, 96]);
    }
}
