use crate::math::tensor::Tensor;
use num::traits::{zero, Float, FromPrimitive, Num, One, Zero};
use rand::Rng;
use std::iter::Sum;
use std::ops::{Add, Mul};

/// Function to perform Hadamard product
pub fn hadamard_product<T>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T>
where
    T: Num + Copy,
{
    assert_eq!(a.shape, b.shape);
    let data = a
        .get_data()
        .iter()
        .zip(b.get_data().iter())
        .map(|(x, y)| *x * *y)
        .collect();
    Tensor::new(data, a.shape.clone())
}

/// Function to perform element-wise addition
pub fn sum<T>(tensor: &Tensor<T>) -> T
where
    T: Num + Copy + Add<T, Output = T> + Zero,
{
    tensor
        .get_data()
        .iter()
        .fold(zero(), |acc, &item| acc + item)
}

/// Function to perform matrix multiplication
pub fn matmul<T>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T>
where
    T: Num + Copy + Add<Output = T> + Mul<Output = T>,
{
    assert_eq!(a.shape.len(), 2);
    assert_eq!(b.shape.len(), 2);

    // matrix dimensions
    let a_rows = a.shape[0];
    let a_cols = a.shape[1];
    // let b_rows = b.shape[0];
    let b_cols = b.shape[1];

    assert_eq!(a.shape[1], b.shape[0]);

    // initialize result vector
    let mut res = Tensor::new(vec![zero(); a_rows * b_cols], vec![a_rows, b_cols]);

    for i in 0..a_rows {
        for j in 0..b_cols {
            for k in 0..a_cols {
                res[&[i, j]] = res[&[i, j]] + a[&[i, k]] * b[&[k, j]];
            }
        }
    }
    res
}

/// Function to compute the magnitude of a vector
pub fn magnitude<T>(vector: &[T]) -> T
where
    T: Float + Sum,
{
    vector.iter().map(|&x| x * x).sum::<T>().sqrt()
}

/// Function to normalize a vector
pub fn normalize<T>(vector: &[T]) -> Vec<T>
where
    T: Float + Sum,
{
    let mag = magnitude(vector);
    vector.iter().map(|&x| x / mag).collect()
}

/// Function to calculate the product of all elements in an array
pub fn mul_vals<T>(arr: &[T]) -> T
where
    T: Mul<Output = T> + One + Copy,
{
    arr.iter().fold(T::one(), |acc, &x| acc * x)
}

/// Function to retrieve a vector of standard distribution random values
pub fn init_random_vec<T>(size: usize) -> Vec<T>
where
    T: Float + FromPrimitive,
{
    let mut rng = rand::thread_rng();
    (0..size).map(|_| T::from_f64(rng.gen()).unwrap()).collect()
}
