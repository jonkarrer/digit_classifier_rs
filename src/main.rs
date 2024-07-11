mod comparing_raw_loss;
mod simple_sgd_model;
mod tensor_tools;

use std::ops::{Add, Mul};

use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::{linear, ops::sigmoid, VarBuilder, VarMap};
use tensor_tools::{stack_tensors_on_axis, transform_dir_into_tensors};

struct Dataset {
    training_inputs: Tensor,
    training_labels: Tensor,
}

struct Linear {
    weights: Tensor,
    bias: Tensor,
}

impl Linear {
    fn forward(&self, inputs: &Tensor) -> Result<Tensor> {
        let x = inputs.matmul(&self.weights)?;
        x.broadcast_add(&self.bias)
    }
}

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;

    // ** 0. Prepare dataset
    // Get the images from the directories and convert them to tensors
    let threes_tensor_list = transform_dir_into_tensors("dataset/training/3");
    let sevens_tensor_list = transform_dir_into_tensors("dataset/training/7");
    let stacked_three_tensor = stack_tensors_on_axis(&threes_tensor_list, 0);
    let stacked_seven_tensor = stack_tensors_on_axis(&sevens_tensor_list, 0);

    // Concat tensors for x axis and reshape to 2-dimensional tensor
    // Rank 2:[[pixel; 784]; 8752]
    let training_set_x = Tensor::cat(&[&stacked_three_tensor, &stacked_seven_tensor], 0)?;
    let training_inputs = training_set_x.reshape(((), 28 * 28))?;

    // Create target tensor using 1 for 3 and 0 for 7
    // Rank 2: [[label]; 8752]
    let mut labels: Vec<f32> = vec![1f32; threes_tensor_list.len()];
    labels.extend(vec![0f32; sevens_tensor_list.len()]);
    let dim_length = labels.len();
    let training_labels = Tensor::from_vec(labels, (dim_length, 1), &device)?;

    let dataset = Dataset {
        training_inputs,
        training_labels,
    };

    // ** 1. Initialize parameters (weights and biases)
    // Weights must be the same shape as one indices tensor of X. In this case X[0] is a 784 item vector, [[0-255; 784]].
    // Rank 2: [[random_num]; 784]
    let weights = Tensor::randn(0f32, 1.0, (28 * 28, 1), &device)?;

    // Bias can just be a random scalar
    // [[random_num]]
    let biases = Tensor::randn(0f32, 1.0, 1, &device)?;

    let model = Linear {
        weights: weights,
        bias: biases,
    };

    // ** 2. Calculate predictions.
    //The root function here is the linear function, y = wx + b,
    // Where w is the weights tensor, x is the input tensor, and b is the bias tensor.
    let _predict_just_one = dataset
        .training_inputs
        .get(0)? // Get first row [0-255; 784]
        .mul(&model.weights.squeeze(1)?)? // Multiply each item by each weight, [0-255; 784] * [random_num; 784]
        .sum_all() // Sum all the items in the new vector [products; 784] = [scalar]
        .add(model.bias.get(0)?)?; // Add the bias to the scalar. scalar + bias = [scalar]

    // We can use matrix multiplication to do the same thing, but on each row of the rank 2 tensors.
    // Weights and inputs are both rank 2, so [[weights]; 784] x [[pixels; 784]; 8752] = [[predictions]; 8752]
    let predictions = model.forward(&dataset.training_inputs)?;

    // ** 3. Calculate loss
    let loss = loss_with_sigmoid(&predictions, &dataset.training_labels)?;

    // ** 4. Optimize loss

    Ok(())
}

fn _loss_no_sigmoid(prediction: &Tensor, target: &Tensor) -> Result<Tensor> {
    target
        .eq(1f32)?
        .where_cond(&Tensor::ones_like(&target)?.sub(&prediction)?, &prediction)?
        .mean(0)
}

fn loss_with_sigmoid(prediction: &Tensor, target: &Tensor) -> Result<Tensor> {
    let prediction = sigmoid(prediction)?;

    target
        .eq(1f32)?
        .where_cond(&Tensor::ones_like(&target)?.sub(&prediction)?, &prediction)?
        .mean(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss() {
        let device = Device::Cpu;
        let t = Tensor::from_vec(vec![1f32, 0f32, 1f32], &[3], &device).unwrap();
        let p = Tensor::from_vec(vec![0.9f32, 0.4f32, 0.2f32], &[3], &device).unwrap();
        let loss = _loss_no_sigmoid(&p, &t).unwrap();
        dbg!(&loss);
        assert!(loss.to_scalar::<f32>().unwrap() == 0.43333333f32);
    }
}
