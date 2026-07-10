use crate::math::tensor::Tensor;
use num::Num;

pub trait Layer<T: Num + Copy> {
    fn forward(&mut self, input_volume: &mut Tensor<T>) -> Tensor<T>;
    fn backward(&mut self, d_out: &mut Tensor<T>) -> Tensor<T>;
}
