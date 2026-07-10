use crate::math::tensor::Tensor;
use num::traits::{Float, FromPrimitive};
use std::iter::Sum;

/// Cross-Entropy loss for multi-class classification
/// Expects softmax probability distribution on forward for numerical stability
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    /// Compute cross-entropy loss
    ///
    /// Args:
    ///   predictions: softmax probabilities [num_classes]
    ///   target: one-hot encoded labels [num_classes]
    ///
    /// Returns:
    ///   loss scalar value
    pub fn forward<T>(predictions: &Tensor<T>, target: &Tensor<T>) -> T
    where
        T: Float + Sum + FromPrimitive,
    {
        let num_classes = predictions.shape()[0];
        assert_eq!(
            predictions.shape(),
            target.shape(),
            "Predictions and target must have same shape"
        );
        assert_eq!(
            predictions.shape().len(),
            1,
            "CrossEntropyLoss expects 1D input"
        );

        let mut loss = T::zero();
        let epsilon = T::from_f64(1e-7).unwrap(); // Numerical stability

        for i in 0..num_classes {
            let pred = predictions[&[i]];
            let targ = target[&[i]];
            // Clamp predictions to avoid log(0)
            let clamped = pred.max(epsilon);
            loss = loss - targ * clamped.ln();
        }

        loss
    }

    /// Compute gradient of cross-entropy loss w.r.t. logits (after softmax)
    ///
    /// For CE loss with softmax:
    ///   d_logits = softmax(logits) - target
    ///
    /// Args:
    ///   predictions: softmax probabilities [num_classes]
    ///   target: one-hot encoded labels [num_classes]
    ///
    /// Returns:
    ///   gradient [num_classes]
    pub fn backward<T>(predictions: &Tensor<T>, target: &Tensor<T>) -> Tensor<T>
    where
        T: Float + Sum + FromPrimitive,
    {
        let num_classes = predictions.shape()[0];
        assert_eq!(
            predictions.shape(),
            target.shape(),
            "Predictions and target must have same shape"
        );

        let mut gradient = Tensor::new(vec![T::zero(); num_classes], vec![num_classes]);

        for i in 0..num_classes {
            gradient[&[i]] = predictions[&[i]] - target[&[i]];
        }

        gradient
    }
}

/// Softmax activation function for final output layer
pub struct Softmax;

impl Softmax {
    /// Apply softmax to logits
    ///
    /// softmax(x)[i] = exp(x[i]) / sum(exp(x))
    ///
    /// Args:
    ///   logits: raw model output [num_classes]
    ///
    /// Returns:
    ///   probabilities: normalized softmax output [num_classes]
    pub fn forward<T>(logits: &Tensor<T>) -> Tensor<T>
    where
        T: Float + Sum + FromPrimitive,
    {
        let num_classes = logits.shape()[0];
        assert_eq!(logits.shape().len(), 1, "Softmax expects 1D input");

        // Numerical stability: subtract max before exp
        let mut max_logit = logits[&[0]];
        for i in 1..num_classes {
            if logits[&[i]] > max_logit {
                max_logit = logits[&[i]];
            }
        }

        let mut exp_logits = vec![T::zero(); num_classes];
        let mut sum_exp = T::zero();

        for i in 0..num_classes {
            exp_logits[i] = (logits[&[i]] - max_logit).exp();
            sum_exp = sum_exp + exp_logits[i];
        }

        let mut softmax_out = vec![T::zero(); num_classes];
        for i in 0..num_classes {
            softmax_out[i] = exp_logits[i] / sum_exp;
        }

        Tensor::new(softmax_out, vec![num_classes])
    }

    /// Backward pass for softmax (used if needed separately)
    /// Note: In practice, we combine softmax + CE loss for more stable gradients
    pub fn backward<T>(softmax_output: &Tensor<T>, d_out: &Tensor<T>) -> Tensor<T>
    where
        T: Float + Sum + FromPrimitive,
    {
        let num_classes = softmax_output.shape()[0];
        assert_eq!(softmax_output.shape(), d_out.shape());

        let mut d_logits = Tensor::new(vec![T::zero(); num_classes], vec![num_classes]);

        // Jacobian of softmax: J[i,j] = softmax[i] * (delta[i,j] - softmax[j])
        for i in 0..num_classes {
            let mut sum = T::zero();
            for j in 0..num_classes {
                let delta = if i == j { T::one() } else { T::zero() };
                let jacobian = softmax_output[&[i]] * (delta - softmax_output[&[j]]);
                sum = sum + jacobian * d_out[&[j]];
            }
            d_logits[&[i]] = sum;
        }

        d_logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let logits = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![3]);
        let softmax = Softmax::forward(&logits);

        // Check shape
        assert_eq!(softmax.shape(), vec![3]);

        // Check sum to 1
        let sum: f32 = softmax.get_data().iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Softmax should sum to 1, got {}",
            sum
        );

        // Check all positive
        for val in softmax.get_data() {
            assert!(*val > 0.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_cross_entropy_loss_perfect_prediction() {
        let predictions = Tensor::new(vec![0.9f32, 0.05, 0.05], vec![3]);
        let target = Tensor::new(vec![1.0f32, 0.0, 0.0], vec![3]);

        let loss = CrossEntropyLoss::forward(&predictions, &target);

        // Should be -log(0.9) ≈ 0.1054
        assert!(
            loss > 0.0 && loss < 0.2,
            "Low loss for good prediction, got {}",
            loss
        );
    }

    #[test]
    fn test_cross_entropy_loss_bad_prediction() {
        let predictions = Tensor::new(vec![0.1f32, 0.1, 0.8], vec![3]);
        let target = Tensor::new(vec![1.0f32, 0.0, 0.0], vec![3]);

        let loss = CrossEntropyLoss::forward(&predictions, &target);

        // Should be -log(0.1) ≈ 2.3026
        assert!(loss > 2.0, "High loss for bad prediction, got {}", loss);
    }

    #[test]
    fn test_cross_entropy_backward() {
        let predictions = Tensor::new(vec![0.7f32, 0.2, 0.1], vec![3]);
        let target = Tensor::new(vec![1.0f32, 0.0, 0.0], vec![3]);

        let gradient = CrossEntropyLoss::backward(&predictions, &target);

        assert_eq!(gradient.shape(), vec![3]);
        // gradient[0] = 0.7 - 1.0 = -0.3
        assert!((gradient[&[0]] - (-0.3)).abs() < 1e-6);
        // gradient[1] = 0.2 - 0.0 = 0.2
        assert!((gradient[&[1]] - 0.2).abs() < 1e-6);
        // gradient[2] = 0.1 - 0.0 = 0.1
        assert!((gradient[&[2]] - 0.1).abs() < 1e-6);
    }
}
