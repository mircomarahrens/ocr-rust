use crate::math::tensor::Tensor;
use num::Num;

pub trait Layer<T: Num + Copy> {
    fn forward(&self, input_volume: &mut Tensor<T>) -> Tensor<T>;
    fn backward(&self, input_volume: &mut Tensor<T>) -> Tensor<T>;
}
