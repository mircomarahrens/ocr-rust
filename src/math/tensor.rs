use core::ops;
use num_traits::Num;

#[derive(Debug, Clone)]
pub struct Tensor<T>
where
    T: Num + Copy,
{
    // The shape of the tensor
    pub shape: Vec<usize>,
    // The rank of the tensor
    pub rank: usize,
    // The strides of the tensor
    strides: Vec<usize>,
    // The data of the tensor
    data: Vec<T>,
}

impl<T> Tensor<T>
where
    T: Num + Copy,
{
    // Constructor
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        let strides = Self::compute_strides(&shape);
        let rank = shape.len();
        Tensor {
            data,
            shape,
            strides,
            rank,
        }
    }

    // Transpose the tensor
    pub fn transpose(mut self) {
        let mut shape = self.shape.clone();
        shape.reverse();
        let strides = Self::compute_strides(&shape);

        self.rank = shape.len();
        self.shape = shape;
        self.strides = strides;
    }

    // Transpose the tensor with a given permutation
    pub fn transpose_with_permutation(mut self, permutation: Vec<usize>) {
        let mut shape = vec![0; self.rank];
        let mut strides = vec![0; self.rank];
        for i in 0..self.rank {
            shape[i] = self.shape[permutation[i]];
            strides[i] = self.strides[permutation[i]];
        }

        self.shape = shape;
        self.strides = strides;
    }

    // Reshape the tensor
    pub fn reshape(&mut self, new_shape: Vec<usize>) {
        assert_eq!(
            self.shape.iter().product::<usize>(),
            new_shape.iter().product::<usize>()
        );
        let strides = Self::compute_strides(&new_shape);
        self.shape = new_shape;
        self.strides = strides;
    }

    // Get the data of the tensor
    pub fn get_data(&self) -> &Vec<T> {
        &self.data
    }

    // Compute the strides given a shape in row-major order
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let d = shape.len();
        let mut stride = 1;
        let mut strides = vec![1; d];
        // Iterate over dimensions in reverse order
        for (dim, &size) in shape.iter().rev().enumerate() {
            strides[d - 1 - dim] = stride;
            stride *= size;
        }
        strides
    }

    // Flatten indices to a single index
    fn flatten(&self, indices: &[usize], strides: &[usize]) -> usize {
        indices.iter().zip(strides).map(|(&i, &s)| i * s).sum()
    }

    // Unflatten a single index to indices
    fn unflatten(&self, index: usize, strides: &[usize]) -> Vec<usize> {
        let mut index = index;
        let mut indices = vec![0; self.rank];
        for i in 0..self.rank {
            indices[i] = index / strides[i];
            index %= strides[i];
        }
        indices
    }

    // Get the shape of the tensor
    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    // Get the rank of the tensor
    pub fn rank(&self) -> usize {
        self.rank
    }

    // Get the strides of the tensor
    pub fn strides(&self) -> Vec<usize> {
        self.strides.clone()
    }

    // Pad the tensor
    pub fn pad(&mut self, pad_spread: usize, pad_value: T)
    where
        T: Clone + Copy,
    {
        let new_shape: Vec<usize> = self.shape.iter().map(|&dim| dim + 2 * pad_spread).collect();
        let mut new_data: Vec<T> = vec![pad_value; new_shape.iter().product()];
        let new_strides = Self::compute_strides(&new_shape);

        for i in 0..self.data.len() {
            let indices = self.unflatten(i, &self.strides);
            let new_indices: Vec<usize> = indices.iter().map(|&idx| idx + pad_spread).collect();
            let new_index = self.flatten(&new_indices, &new_strides);
            new_data[new_index] = self.data[i];
        }

        self.data = new_data;
        self.shape = new_shape;
        self.strides = new_strides;
    }

    // Get the maximum value in the tensor
    pub fn max(&self) -> T
    where
        T: PartialOrd,
    {
        let mut max_value = self.data[0];
        for &value in &self.data {
            if value > max_value {
                max_value = value;
            }
        }
        max_value
    }

    // Slice the tensor
    // TODO: Returns a new tensor. Maybe a subview would be better for performance.
    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Tensor<T> {
        assert_eq!(ranges.len(), self.rank);

        let new_shape: Vec<usize> = ranges.iter().map(|r| r.end - r.start).collect();
        let new_strides = Self::compute_strides(&new_shape);
        let new_data_size: usize = new_shape.iter().product();
        let mut new_data: Vec<T> = vec![T::zero(); new_data_size];

        for (i, elem) in new_data.iter_mut().enumerate() {
            let new_indices = self.unflatten(i, &new_strides);
            let original_indices: Vec<usize> = new_indices
                .iter()
                .enumerate()
                .map(|(dim, &idx)| idx + ranges[dim].start)
                .collect();
            let original_index = self.flatten(&original_indices, &self.strides);
            *elem = self.data[original_index];
        }

        Tensor {
            shape: new_shape,
            rank: self.rank,
            strides: new_strides,
            data: new_data,
        }
    }
}

impl<T> ops::Index<&[usize]> for Tensor<T>
where
    T: Num + Copy,
{
    type Output = T;

    fn index(&self, indices: &[usize]) -> &T {
        let index = self.flatten(indices, &self.strides);
        &self.data[index]
    }
}

impl<T> ops::IndexMut<&[usize]> for Tensor<T>
where
    T: Num + Copy,
{
    fn index_mut(&mut self, indices: &[usize]) -> &mut T {
        let index = self.flatten(indices, &self.strides);
        &mut self.data[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_indexing() {
        let shape = vec![2, 3, 4];
        let data_size = shape.iter().product();
        let data = (0..data_size).map(|v| v as i32).collect::<Vec<i32>>();

        let tensor = Tensor::new(data, shape);

        assert_eq!(tensor[&[0, 0, 0]], 0);
        assert_eq!(tensor[&[1, 2, 3]], 23);
    }

    #[test]
    fn test_tensor_slicing() {
        let shape = vec![2, 3];
        let data_size = shape.iter().product();
        let data = (1..=data_size).map(|v| v as i32).collect::<Vec<i32>>();

        let tensor = Tensor::new(data, shape);

        // slice all rows, last two columns: cols 1..3
        let view = tensor.slice(&[0..2, 1..3]);
        assert_eq!(view.shape(), &[2, 2]);

        // original layout:
        // [ [1,2,3],
        //   [4,5,6] ]
        assert_eq!(view[&[0, 0]], 2);
        assert_eq!(view[&[0, 1]], 3);
        assert_eq!(view[&[1, 0]], 5);
        assert_eq!(view[&[1, 1]], 6);
    }
}
