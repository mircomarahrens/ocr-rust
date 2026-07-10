use crate::math::tensor::Tensor;
use num::traits::{Float, FromPrimitive};

/// Stochastic Gradient Descent optimizer
pub struct SGD<T>
where
    T: Float,
{
    pub learning_rate: T,
    pub momentum: Option<T>,
    velocity_weights: Option<Tensor<T>>,
    velocity_bias: Option<Tensor<T>>,
}

impl<T> SGD<T>
where
    T: Float + FromPrimitive,
{
    /// Create a new SGD optimizer without momentum
    pub fn new(learning_rate: T) -> Self {
        SGD {
            learning_rate,
            momentum: None,
            velocity_weights: None,
            velocity_bias: None,
        }
    }

    /// Create SGD optimizer with momentum
    pub fn with_momentum(learning_rate: T, momentum: T) -> Self {
        SGD {
            learning_rate,
            momentum: Some(momentum),
            velocity_weights: None,
            velocity_bias: None,
        }
    }

    /// Update weights using gradient
    /// w = w - learning_rate * d_weights
    pub fn update_weights(&mut self, weights: &mut Tensor<T>, d_weights: &Tensor<T>) {
        let shape = weights.shape();

        if let Some(m) = self.momentum {
            // Initialize velocity if needed
            if self.velocity_weights.is_none() {
                self.velocity_weights = Some(Tensor::new(
                    vec![T::zero(); d_weights.get_data().len()],
                    shape.clone(),
                ));
            }

            let velocity = self.velocity_weights.as_mut().unwrap();
            // v = m * v + (1 - m) * d_weights
            // Actually: v = m * v - learning_rate * d_weights
            for (i, dw) in d_weights.get_data().iter().enumerate() {
                velocity.get_data_mut()[i] = m * velocity.get_data()[i] - self.learning_rate * *dw;
                weights.get_data_mut()[i] = weights.get_data()[i] + velocity.get_data()[i];
            }
        } else {
            // Standard SGD: w = w - learning_rate * d_weights
            for (i, dw) in d_weights.get_data().iter().enumerate() {
                weights.get_data_mut()[i] = weights.get_data()[i] - self.learning_rate * *dw;
            }
        }
    }

    /// Update bias using gradient
    /// b = b - learning_rate * d_bias
    pub fn update_bias(&mut self, bias: &mut Tensor<T>, d_bias: &Tensor<T>) {
        let shape = bias.shape();

        if let Some(m) = self.momentum {
            // Initialize velocity if needed
            if self.velocity_bias.is_none() {
                self.velocity_bias = Some(Tensor::new(
                    vec![T::zero(); d_bias.get_data().len()],
                    shape.clone(),
                ));
            }

            let velocity = self.velocity_bias.as_mut().unwrap();
            for (i, db) in d_bias.get_data().iter().enumerate() {
                velocity.get_data_mut()[i] = m * velocity.get_data()[i] - self.learning_rate * *db;
                bias.get_data_mut()[i] = bias.get_data()[i] + velocity.get_data()[i];
            }
        } else {
            for (i, db) in d_bias.get_data().iter().enumerate() {
                bias.get_data_mut()[i] = bias.get_data()[i] - self.learning_rate * *db;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_update_weights() {
        let mut sgd: SGD<f32> = SGD::new(0.1);
        let mut weights = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![3]);
        let gradients = Tensor::new(vec![0.1f32, 0.2, 0.3], vec![3]);

        sgd.update_weights(&mut weights, &gradients);

        // w = w - lr * dw
        // w[0] = 1.0 - 0.1 * 0.1 = 0.99
        assert!((weights[&[0]] - 0.99).abs() < 1e-6);
        // w[1] = 2.0 - 0.1 * 0.2 = 1.98
        assert!((weights[&[1]] - 1.98).abs() < 1e-6);
        // w[2] = 3.0 - 0.1 * 0.3 = 2.97
        assert!((weights[&[2]] - 2.97).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_update_bias() {
        let mut sgd: SGD<f32> = SGD::new(0.01);
        let mut bias = Tensor::new(vec![0.5f32, -0.5], vec![2]);
        let gradients = Tensor::new(vec![0.1f32, -0.1], vec![2]);

        sgd.update_bias(&mut bias, &gradients);

        // b = b - lr * db
        // b[0] = 0.5 - 0.01 * 0.1 = 0.499
        assert!((bias[&[0]] - 0.499).abs() < 1e-6);
        // b[1] = -0.5 - 0.01 * (-0.1) = -0.499
        assert!((bias[&[1]] - (-0.499)).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_with_momentum() {
        let mut sgd: SGD<f32> = SGD::with_momentum(0.1, 0.9);
        let mut weights = Tensor::new(vec![1.0f32], vec![1]);
        let gradients1 = Tensor::new(vec![1.0f32], vec![1]);

        sgd.update_weights(&mut weights, &gradients1);
        // First update: v = 0.9 * 0 - 0.1 * 1.0 = -0.1
        // w = 1.0 + (-0.1) = 0.9
        assert!((weights[&[0]] - 0.9).abs() < 1e-6);

        // Second update with same gradient
        sgd.update_weights(&mut weights, &gradients1);
        // v = 0.9 * (-0.1) - 0.1 * 1.0 = -0.09 - 0.1 = -0.19
        // w = 0.9 + (-0.19) = 0.71
        assert!((weights[&[0]] - 0.71).abs() < 1e-6);
    }
}
