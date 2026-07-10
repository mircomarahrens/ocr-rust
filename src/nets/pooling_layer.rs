use crate::nets::layers::Layer;

use crate::math::linalg::mul_vals;
use num::traits::{Float, FromPrimitive};
use std::iter::Sum;

use crate::math::tensor::Tensor;

/// Pooling layer for a neural network.
pub struct PoolingLayer {
    spatial_extent: Vec<usize>, // Spatial extent of the pooling layer, F
    stride: usize,              // Stride for the pooling layer, S
    argmax_cache: Vec<(usize, usize)>, // (w_in, h_in) per (output_idx * depth + depth_idx)
    input_shape_cache: Vec<usize>, // input shape stored during forward for backward
}

impl PoolingLayer {
    /// Constructor for the pooling layer.
    pub fn new(spatial_extent: Vec<usize>, stride: usize) -> Self {
        PoolingLayer {
            spatial_extent,
            stride,
            argmax_cache: Vec::new(),
            input_shape_cache: Vec::new(),
        }
    }
}

impl<T> Layer<T> for PoolingLayer
where
    T: Float + Sum + FromPrimitive,
{
    fn forward(&mut self, input_volume: &mut Tensor<T>) -> Tensor<T> {
        self.input_shape_cache = input_volume.shape();

        let f_w = self.spatial_extent[0];
        let f_h = self.spatial_extent[1];
        let d = input_volume.shape()[2];

        let w_2 = (input_volume.shape()[0] - f_w) / self.stride + 1;
        let h_2 = (input_volume.shape()[1] - f_h) / self.stride + 1;

        let output_shape = vec![w_2, h_2, d];
        let data_size = mul_vals(&output_shape);
        let data = vec![T::zero(); data_size];
        let mut output_volume = Tensor::new(data, output_shape);

        let number_of_regions = w_2 * h_2;
        self.argmax_cache = vec![(0, 0); number_of_regions * d];

        for idx in 0..number_of_regions {
            let w_out = idx / h_2;
            let h_out = idx % h_2;

            let w_start = w_out * self.stride;
            let h_start = h_out * self.stride;

            for depth in 0..d {
                let mut max_value = T::neg_infinity();
                let mut max_w = w_start;
                let mut max_h = h_start;

                for fw in 0..f_w {
                    for fh in 0..f_h {
                        let w_in = w_start + fw;
                        let h_in = h_start + fh;

                        let value = input_volume[&[w_in, h_in, depth]];
                        if value > max_value {
                            max_value = value;
                            max_w = w_in;
                            max_h = h_in;
                        }
                    }
                }

                output_volume[&[w_out, h_out, depth]] = max_value;
                self.argmax_cache[idx * d + depth] = (max_w, max_h);
            }
        }

        output_volume
    }

    fn backward(&mut self, d_out: &mut Tensor<T>) -> Tensor<T> {
        let input_shape = self.input_shape_cache.clone();
        let data_size: usize = input_shape.iter().product();
        let mut d_input = Tensor::new(vec![T::zero(); data_size], input_shape);

        let d = d_out.shape()[2];
        let w_2 = d_out.shape()[0];
        let h_2 = d_out.shape()[1];

        for idx in 0..(w_2 * h_2) {
            let w_out = idx / h_2;
            let h_out = idx % h_2;

            for depth in 0..d {
                let (w_in, h_in) = self.argmax_cache[idx * d + depth];
                d_input[&[w_in, h_in, depth]] =
                    d_input[&[w_in, h_in, depth]] + d_out[&[w_out, h_out, depth]];
            }
        }

        d_input
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward() {
        // Test the forward pass of the pooling layer
        let spatial_extent = vec![2, 2];
        let stride = 2;

        let mut pooling_layer = PoolingLayer::new(spatial_extent.clone(), stride);

        let w_1 = 55;
        let h_1 = 55;
        let d_1 = 96;

        let shape = vec![w_1, h_1, d_1];
        let data_size = mul_vals(&shape);
        let data = (1..data_size + 1).map(|v| v as f32).collect::<Vec<f32>>();
        let mut input_volume = Tensor::new(data, shape.clone());

        let output_volume = pooling_layer.forward(&mut input_volume);

        let w_2 = (w_1 - spatial_extent[0]) / stride + 1;
        let h_2 = (h_1 - spatial_extent[1]) / stride + 1;
        let d_2 = d_1;

        // TODO: provide a general way to verify correctness, not just specific values

        // println!();
        // for i in 0..8 {
        //     for j in 0..8 {
        //         print!("{:4} ", input_volume[&[i, j, 0]]);
        //     }
        //     println!();
        // }

        assert_eq!(output_volume.shape(), vec![w_2, h_2, d_2]);

        // println!();
        // for i in 0..2 {
        //     for j in 0..2 {
        //         print!("{:4} ", input_volume[&[i, j, 0]]);
        //     }
        //     println!();
        // }

        assert_eq!(output_volume[&[0, 0, 0]], 5377.0);

        // println!();
        // for i in 0..2 {
        //     for j in 2..4 {
        //         print!("{:4} ", input_volume[&[i, j, 0]]);
        //     }
        //     println!();
        // }

        assert_eq!(output_volume[&[0, 1, 0]], 5569.0);

        // println!();
        // for i in 2..4 {
        //     for j in 0..2 {
        //         print!("{:4} ", input_volume[&[i, j, 0]]);
        //     }
        //     println!();
        // }

        assert_eq!(output_volume[&[1, 0, 0]], 15937.0);

        // println!();
        // for i in 2..4 {
        //     for j in 2..4 {
        //         print!("{:4} ", input_volume[&[i, j, 0]]);
        //     }
        //     println!();
        // }

        assert_eq!(output_volume[&[1, 1, 0]], 16129.0);

        // println!();
    }

    #[test]
    fn test_backward() {
        // Each output cell must route its gradient to exactly the argmax input position.
        let spatial_extent = vec![2, 2];
        let stride = 2;

        let mut pool = PoolingLayer::new(spatial_extent, stride);

        let input_shape = vec![4, 4, 1];
        let data_size = mul_vals(&input_shape);
        // values 1..=16 so the max of each 2x2 region is always the bottom-right element
        let data: Vec<f32> = (1..=data_size).map(|v| v as f32).collect();
        let mut input = Tensor::new(data, input_shape.clone());

        let output = pool.forward(&mut input);

        let d_out_shape = output.shape();
        let d_out_size: usize = d_out_shape.iter().product();
        let mut d_out = Tensor::new(vec![1.0f32; d_out_size], d_out_shape);

        let d_input = pool.backward(&mut d_out);

        assert_eq!(d_input.shape(), input_shape);
        // There are 4 output cells each with gradient 1.0 → total gradient must be 4.0
        let total: f32 = d_input.get_data().iter().sum();
        assert!((total - 4.0).abs() < 1e-6);
    }
}
