use crate::math::linalg::mul_vals;
use crate::math::tensor::Tensor;
use crate::nets::layers::Layer;
use num::traits::{Float, FromPrimitive};
use std::iter::Sum;

/// Flatten layer: reshapes [W, H, D] → [W*H*D]
#[derive(Clone, Debug)]
pub struct FlattenLayer {
    original_shape: Vec<usize>,
}

impl FlattenLayer {
    pub fn new() -> Self {
        FlattenLayer {
            original_shape: Vec::new(),
        }
    }
}

impl Default for FlattenLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Layer<T> for FlattenLayer
where
    T: Float + Sum + FromPrimitive,
{
    fn forward(&mut self, input_volume: &mut Tensor<T>) -> Tensor<T> {
        self.original_shape = input_volume.shape();
        let flat_size = mul_vals(&self.original_shape);
        let data = input_volume.get_data().clone();
        Tensor::new(data, vec![flat_size])
    }

    fn backward(&mut self, d_out: &mut Tensor<T>) -> Tensor<T> {
        let flat_data = d_out.get_data().clone();
        Tensor::new(flat_data, self.original_shape.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatten_forward() {
        let mut flatten: FlattenLayer = FlattenLayer::new();
        let mut input = Tensor::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3, 1],
        );

        let output = flatten.forward(&mut input);

        assert_eq!(output.shape(), vec![9]);
        assert_eq!(
            output.get_data(),
            &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn test_flatten_backward() {
        let mut flatten: FlattenLayer = FlattenLayer::new();
        let mut input = Tensor::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3, 1],
        );

        let _ = flatten.forward(&mut input);

        let mut d_out = Tensor::new(vec![1.0f32; 9], vec![9]);
        let d_input = flatten.backward(&mut d_out);

        assert_eq!(d_input.shape(), vec![3, 3, 1]);
        assert_eq!(d_input.get_data(), &vec![1.0; 9]);
    }
}
