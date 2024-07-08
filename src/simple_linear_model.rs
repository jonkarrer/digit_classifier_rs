mod comparing_raw_loss;
mod stochastic_gradient_decent;
mod tensor_tools;

use std::ops::{Add, Mul};

use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{linear, Linear, VarBuilder, VarMap};
use tensor_tools::{stack_tensors_on_axis, transform_dir_into_tensors};

struct Dataset {
    training_examples: Tensor,
    training_results: Tensor,
}

struct Model {
    lnn_one: Linear,
    lnn_two: Linear,
    lnn_three: Linear,
}

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;

    // Get the images from the directories and convert them to tensors
    let threes_tensor_list = transform_dir_into_tensors("dataset/training/3");
    let sevens_tensor_list = transform_dir_into_tensors("dataset/training/7");
    dbg!(threes_tensor_list.len(), sevens_tensor_list.len());
    dbg!(threes_tensor_list[0].dims(), sevens_tensor_list[0].dims());

    // Stack tensors on one axis
    let stacked_three_tensor = stack_tensors_on_axis(&threes_tensor_list, 0);
    let stacked_seven_tensor = stack_tensors_on_axis(&sevens_tensor_list, 0);
    dbg!(stacked_three_tensor.shape(), stacked_seven_tensor.shape());

    // Concat tensors for x axis
    let training_set_x = Tensor::cat(&[&stacked_three_tensor, &stacked_seven_tensor], 0)?;
    dbg!(training_set_x.shape());

    // Reshape n-dimensional tensor to 2-dimensional tensor
    let training_set_x = training_set_x.reshape(((), 28 * 28))?;
    dbg!(training_set_x.shape());

    // Create target tensor using 1 for 3 and 0 for 7
    let mut target_threes: Vec<u8> = vec![1; threes_tensor_list.len()];
    let target_sevens: Vec<u8> = vec![0; sevens_tensor_list.len()];
    target_threes.extend(target_sevens);
    let first_dimension = target_threes.len();

    let training_set_y = Tensor::from_vec(target_threes, &[first_dimension, 1], &device)?;
    dbg!(training_set_y.shape());

    // Prepare input tensors
    let dataset = Dataset {
        training_examples: training_set_x,
        training_results: training_set_y,
    };

    // 1. Initialize parameters (weights and biases)
    // Weights must be the same shape as one index tensor of X. In this case 1 index of X is a 28x28 image tensor. The whole X axis is thousands of images, but we just want to match the size of one.
    let weights = Var::from_tensor(&Tensor::randn(1.0 as f32, 1.0 as f32, 28 * 28, &device)?)?;
    // Bias can just be a random scalar
    let biases = Var::from_tensor(&Tensor::randn(0.0 as f32, 1.0 as f32, 1, &device)?)?;

    // 2. Calculate predictions. The root function here is the linear function, y = wx + b, where w is the weights tensor, x is the input tensor, and b is the bias tensor.
    // The wx multiplication is a matrix multiplication. It's a 28x28 matrix times 28x28 matrix.
    let prediction = training_set_x
        .get_on_dim(0, 0)?
        .mul(weights.as_tensor())?
        .sum(0)
        .add(biases.get(0)?)?;
    dbg!(&prediction);

    Ok(())
}
