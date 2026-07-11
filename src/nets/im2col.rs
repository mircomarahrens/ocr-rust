use num::traits::{zero, Float, FromPrimitive};
use std::iter::Sum;

use crate::math::linalg::{init_random_vec, mul_vals};
use crate::math::tensor::Tensor;
use rand::distributions::{Distribution, Standard};

pub fn calc_w_row<T>(num_filters: usize, filter_size: usize) -> Tensor<T>
where
    T: Float + Sum + FromPrimitive,
    Standard: Distribution<T>,
{
    let w_row_shape = vec![num_filters, filter_size];
    let mut w_row_data = vec![zero(); mul_vals(&w_row_shape)];

    for i in 0..num_filters {
        let weights: Vec<T> = init_random_vec(filter_size);
        for (j, e) in weights.iter().enumerate() {
            w_row_data[j + i * filter_size] = *e;
        }
    }
    Tensor::new(w_row_data, w_row_shape)
}

pub fn calc_col_shape<T>(
    input_volume: &Tensor<T>,
    spatial_extent: &[usize],
    stride: usize,
) -> Vec<usize>
where
    T: Float,
{
    let input_shape = input_volume.shape();
    assert!(
        input_shape.len() == 3 && spatial_extent.len() == 3,
        "Convolution expects rank-3 input and filter shapes [W, H, D]."
    );
    assert_eq!(
        input_shape[2], spatial_extent[2],
        "Input depth must match filter depth."
    );

    let w_divd = input_shape[0] - spatial_extent[0];
    let h_divd = input_shape[1] - spatial_extent[1];
    assert!(
        w_divd % stride == 0 && h_divd % stride == 0,
        "Invalid input for spatial extent or stride."
    );

    vec![w_divd / stride + 1, h_divd / stride + 1]
}

pub fn calc_x_col<T>(
    input_volume: &Tensor<T>,
    col_shape: &[usize],
    filter_size: usize,
    spatial_extent: &[usize],
    stride: usize,
) -> Tensor<T>
where
    T: Float + Sum,
{
    let width = col_shape[0];
    let height = col_shape[1];

    let x_col_size = width * height;
    let x_col_shape = vec![filter_size, x_col_size];
    let mut x_col_data = vec![T::zero(); filter_size * x_col_size];

    let input_data = input_volume.get_data();
    let input_shape = input_volume.shape();
    let in_h = input_shape[1];
    let in_d = input_shape[2];

    for i in 0..width {
        let x = i * stride;
        for j in 0..height {
            let y = j * stride;
            let iy = i * height + j;

            for k in 0..spatial_extent[0] {
                let xp = x + k;
                let xp_offset = xp * in_h * in_d;
                for l in 0..spatial_extent[1] {
                    let yp = y + l;
                    let yp_offset = yp * in_d;
                    for c in 0..spatial_extent[2] {
                        let ix = (k * spatial_extent[1] + l) * spatial_extent[2] + c;
                        x_col_data[ix * x_col_size + iy] = input_data[xp_offset + yp_offset + c];
                    }
                }
            }
        }
    }

    Tensor::new(x_col_data, x_col_shape)
}

/// Reverse of calc_x_col: accumulate column-format gradients back into spatial volume.
/// x_col:        [filter_size, out_w * out_h]
/// padded_shape: [W+2P, H+2P, D]  (padded input dimensions)
/// Returns d_x:  [W+2P, H+2P, D]
pub fn col2im<T>(
    x_col: &Tensor<T>,
    padded_shape: &[usize],
    spatial_extent: &[usize],
    stride: usize,
    out_w: usize,
    out_h: usize,
) -> Tensor<T>
where
    T: Float + Sum,
{
    let data_size: usize = padded_shape.iter().product();
    let mut dx_data = vec![T::zero(); data_size];

    let x_col_data = x_col.get_data();
    let x_col_shape = x_col.shape();
    let x_col_size = x_col_shape[1]; // out_w * out_h

    let in_h = padded_shape[1];
    let in_d = padded_shape[2];

    for i in 0..out_w {
        let x = i * stride;
        for j in 0..out_h {
            let y = j * stride;
            let iy = i * out_h + j;
            for k in 0..spatial_extent[0] {
                let xp = x + k;
                let xp_offset = xp * in_h * in_d;
                for l in 0..spatial_extent[1] {
                    let yp = y + l;
                    let yp_offset = yp * in_d;
                    for c in 0..spatial_extent[2] {
                        let ix = (k * spatial_extent[1] + l) * spatial_extent[2] + c;
                        dx_data[xp_offset + yp_offset + c] = dx_data[xp_offset + yp_offset + c] + x_col_data[ix * x_col_size + iy];
                    }
                }
            }
        }
    }

    Tensor::new(dx_data, padded_shape.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calc_w_row() {
        let num_filters = 96;
        let filter_size = 11 * 11 * 3;
        let w_row = calc_w_row::<f32>(num_filters, filter_size);

        println!("w_row shape: {:?}", w_row.shape());

        assert_eq!(w_row.shape(), vec![num_filters, filter_size]);
    }

    #[test]
    fn test_calc_col_shape() {
        // dummy image
        let shape = vec![227, 227, 3];
        let input_size = 227 * 227 * 3;
        let data: Vec<f32> = (1..input_size + 1).map(|v| v as f32).collect();

        // specify input volume
        let input_volume = Tensor::new(data, shape.clone());

        println!("Input volume shape: {:?}", input_volume.shape());

        let spatial_extent = vec![11, 11, 3];
        let stride = 4;
        let col_shape = calc_col_shape(&input_volume, &spatial_extent, stride);

        println!("col_shape: {:?}", col_shape);

        assert_eq!(col_shape, vec![55, 55]);
    }

    #[test]
    fn test_calc_x_col() {
        // dummy image
        let shape = vec![227, 227, 3];
        let input_size = 227 * 227 * 3;
        let data: Vec<f32> = (1..input_size + 1).map(|v| v as f32).collect();

        // specify input volume
        let input_volume = Tensor::new(data, shape.clone());

        println!("Input volume shape: {:?}", input_volume.shape());

        let filter_size = 11 * 11 * 3;
        let stride = 4;

        let spatial_extent = vec![11, 11, 3];
        let col_shape = calc_col_shape(&input_volume, &spatial_extent, stride);

        println!("col_shape: {:?}", col_shape);

        let x_col = calc_x_col(
            &input_volume,
            &col_shape,
            filter_size,
            &spatial_extent,
            stride,
        );

        println!("x_col shape: {:?}", x_col.shape());

        assert_eq!(x_col.shape(), vec![filter_size, mul_vals(&col_shape)]);
    }
}
