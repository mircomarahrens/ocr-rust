use crate::nets::layers::Layer;

use num::traits::{Float, FromPrimitive};
use std::iter::Sum;

use crate::math::tensor::Tensor;

/// Pooling layer for a neural network.
pub struct PoolingLayer {
    spatial_extent: Vec<usize>, // Spatial extent of the pooling layer, F
    stride: usize,              // Stride for the pooling layer, S
}

impl PoolingLayer {
    /// Constructor for the pooling layer.
    pub fn new(spatial_extent: Vec<usize>, stride: usize) -> Self {
        PoolingLayer {
            spatial_extent,
            stride,
        }
    }
}

impl<T> Layer<T> for PoolingLayer
where
    T: Float + Sum + FromPrimitive,
{
    fn forward(&self, input_volume: &mut Tensor<T>) -> Tensor<T> {
        // TODO
        Tensor::new(vec![], vec![])
    }

    fn backward(&self, input_volume: &mut Tensor<T>) -> Tensor<T> {
        // TODO
        Tensor::new(vec![], vec![])
    }
}
