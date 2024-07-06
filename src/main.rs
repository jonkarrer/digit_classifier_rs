mod comparing_raw_loss;
mod stochastic_gradient_decent;
mod tensor_tools;

use candle_core::{Device, Tensor};
use candle_nn::{linear, Linear, VarBuilder};
use tensor_tools::{stack_tensors_on_axis, transform_dir_into_tensors};

struct Dataset {
    training_examples: Tensor,
    training_results: Tensor,
    validation_examples: Tensor,
    validation_results: Tensor,
}

struct Model {
    lnn_one: Linear,
    lnn_two: Linear,
    lnn_three: Linear,
}

fn main() -> anyhow::Result<()> {
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

    let training_set_y = Tensor::from_vec(target_threes, &[first_dimension, 1], &Device::Cpu)?;
    dbg!(training_set_y.shape());

    Ok(())
}
