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

/// Function to find the maximum value in a tensor
pub fn max<T>(a: &Tensor<T>) -> T
where
    T: Num + Float + Copy,
{
    let mut max_val = a[&[0]];
    for &val in a.get_data().iter() {
        if val > max_val {
            max_val = val;
        }
    }
    max_val
}

/// Function to find the minimum value in a tensor
pub fn min<T>(a: &Tensor<T>) -> T
where
    T: Num + Float + Copy,
{
    let mut min_val = a[&[0]];
    for &val in a.get_data().iter() {
        if val < min_val {
            min_val = val;
        }
    }

    min_val
}

/// Compute a^T @ b  where a is [m, k] and b is [m, n] — returns [k, n]
pub fn matmul_transpose_a<T>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T>
where
    T: Num + Copy + Add<Output = T> + Mul<Output = T>,
{
    assert_eq!(a.shape.len(), 2);
    assert_eq!(b.shape.len(), 2);
    assert_eq!(
        a.shape[0], b.shape[0],
        "matmul_transpose_a: row count mismatch"
    );

    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];

    let mut res = Tensor::new(vec![zero(); k * n], vec![k, n]);
    for i in 0..k {
        for j in 0..n {
            for p in 0..m {
                res[&[i, j]] = res[&[i, j]] + a[&[p, i]] * b[&[p, j]];
            }
        }
    }
    res
}

/// Compute a @ b^T  where a is [m, k] and b is [n, k] — returns [m, n]
pub fn matmul_transpose_b<T>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T>
where
    T: Num + Copy + Add<Output = T> + Mul<Output = T>,
{
    assert_eq!(a.shape.len(), 2);
    assert_eq!(b.shape.len(), 2);
    assert_eq!(
        a.shape[1], b.shape[1],
        "matmul_transpose_b: column count mismatch"
    );

    let m = a.shape[0];
    let n = b.shape[0];
    let k = a.shape[1];

    let mut res = Tensor::new(vec![zero(); m * n], vec![m, n]);
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                res[&[i, j]] = res[&[i, j]] + a[&[i, p]] * b[&[j, p]];
            }
        }
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mul_vals() {
        let arr = vec![2, 3, 4];
        let result = mul_vals(&arr);
        assert_eq!(result, 24);
    }

    #[test]
    fn test_magnitude() {
        let vector = vec![3.0, 4.0];
        let result = magnitude(&vector);
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_normalize() {
        let vector = vec![3.0, 4.0];
        let result = normalize(&vector);
        assert!((result[0] - 0.6).abs() < 1e-6);
        assert!((result[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_hadamard_product() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        let b = Tensor::new(vec![5, 6, 7, 8], vec![2, 2]);
        let result = hadamard_product(&a, &b);
        assert_eq!(result.get_data(), &vec![5, 12, 21, 32]);
    }

    #[test]
    fn test_sum() {
        let tensor = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        let result = sum(&tensor);
        assert_eq!(result, 10);
    }

    #[test]
    fn test_max() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 5.0], vec![2, 2]);
        let result = max(&tensor);
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_min() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 5.0], vec![2, 2]);
        let result = min(&tensor);
        assert_eq!(result, 1.0);
    }
}
