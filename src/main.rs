mod comparing_raw_loss;
mod simple_linear_model;
mod simple_sgd_model;
mod tensor_tools;

use std::ops::{Add, Mul};

use candle_core::{shape, DType, Device, Result, Shape, Tensor};
use candle_nn::{ops::sigmoid, VarBuilder, VarMap};
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
        dbg!(&inputs, &inputs.shape());
        let x = inputs.matmul(&self.weights)?;
        x.broadcast_add(&self.bias)
    }
}

struct Model {
    layer_one: Linear,
    layer_two: Linear,
}

impl Model {
    fn forward(&self, images: &Tensor) -> Result<Tensor> {
        let x = self.layer_one.forward(images)?;
        let x = x.relu()?;
        let y = self.layer_two.forward(&x);
        y
    }
}

fn main() -> anyhow::Result<()> {
    let device = Device::cuda_if_available(0)?;

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
    // Pass them into a layer one
    let weights = init_weights(&device, (28 * 28, 100).into())?;
    let bias = init_bias(&device, (100,).into())?;
    let layer_one = Linear { weights, bias };

    // Pass another set into layer two
    let weights = init_weights(&device, (100, 1).into())?;
    let bias = init_bias(&device, (1,).into())?;
    let layer_two = Linear { weights, bias };

    // Create the model
    let model = Model {
        layer_one,
        layer_two,
    };

    // ** 2. Calculate predictions.
    let predictions = model.forward(&dataset.training_inputs)?;

    // ** 3. Calculate loss
    let loss = loss_with_sigmoid(&predictions, &dataset.training_labels)?;
    dbg!(loss);

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

fn init_weights(device: &Device, shape: Shape) -> Result<Tensor> {
    Tensor::randn(0f32, 1.0, shape, device)
}

fn init_bias(device: &Device, shape: Shape) -> Result<Tensor> {
    Tensor::randn(0f32, 1.0, shape, device)
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
