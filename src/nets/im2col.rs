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
            w_row_data[j + i * num_filters] = *e;
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
    let mut col_shape = Vec::new();
    let input_shape = input_volume.shape();

    for (i, e) in input_shape.iter().enumerate() {
        let divd = e - spatial_extent[i];
        let divs = stride;
        assert!(
            divd % divs == 0,
            "Invalid input for spatial extent or stride."
        );
        let res = divd / divs + 1;
        col_shape.push(res);
    }
    col_shape
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

    let x_col_size = mul_vals(col_shape);
    let x_col_shape = vec![filter_size, x_col_size];
    let x_col_data = vec![zero(); mul_vals(&x_col_shape)];

    // TODO this is a bit ugly, maybe there is a better way to do this
    // calculate the x_col
    let mut x_col = Tensor::new(x_col_data, x_col_shape);
    for i in 0..width {
        // move forward in x direction
        let x = i * stride;
        for j in 0..height {
            // move forward in y direction
            let y = j * stride;

            // get patch and assign to x_col
            for k in 0..spatial_extent[0] {
                // traverse filter spread at correct position in input
                let xp = x + k;
                for l in 0..spatial_extent[1] {
                    let yp = y + l;

                    // transform k, l to single index -> x_col x coordinate
                    let ix = k * spatial_extent[0] + l;
                    // transform i, j to single index -> x_col y coordinate
                    let iy = i * width + j;

                    x_col[&[ix, iy]] = input_volume[&[xp, yp]];
                }
            }
        }
    }
    x_col
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

        assert_eq!(col_shape, vec![55, 55, 1]);
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
