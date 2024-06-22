use candle_core::Tensor;
use candle_nn::loss::mse;
use std::{fs, path::Path};

fn dir_walk(dir_path: &str) -> anyhow::Result<Vec<String>> {
    Ok(fs::read_dir(Path::new(dir_path))?
        .map(|entry| {
            entry
                .expect("Failed to read entry")
                .path()
                .to_str()
                .expect("Failed to convert path to str")
                .to_owned()
        })
        .collect())
}

fn tensor_from_image(img_path: &str) -> anyhow::Result<Tensor> {
    let img = image::open(img_path)?;
    let img_ten = Tensor::from_raw_buffer(
        img.as_bytes(),
        candle_core::DType::U8,
        &[28, 28],
        &candle_core::Device::Cpu,
    )?;
    Ok(img_ten)
}

fn transform_dir_into_tensors(dir_path: &str) -> Vec<Tensor> {
    dir_walk(dir_path)
        .expect("Failed to read directory")
        .into_iter()
        .map(|path| {
            tensor_from_image(&path)
                .expect("Failed to convert image")
                .to_dtype(candle_core::DType::F32)
                .expect("Failed to convert tensor to f32")
        })
        .collect()
}

fn stack_tensors_on_axis(tensor_list: &[Tensor], axis: usize) -> Tensor {
    Tensor::stack(tensor_list, axis)
        .expect("Failed to stack tensors on dimension")
        .to_dtype(candle_core::DType::F32)
        .expect("Failed to convert tensor to f32")
}

fn average_tensor_on_axis(tensor: &Tensor, axis: usize) -> Tensor {
    tensor.mean(axis).expect("Failed to calculate mean")
}

fn mnist_distance(a: &Tensor, b: &Tensor) -> f32 {
    let result = a.broadcast_sub(b).expect("Failed to broadcast subtraction");
    let abs_result = result.abs().expect("Failed to calculate absolute value");

    abs_result
        .mean_all()
        .expect("Failed to calculate mean")
        .to_vec0::<f32>()
        .unwrap()
}

fn is_a_three(test_tensor: &Tensor, ideal_tensor: &Tensor, threshhold_tensor: &Tensor) -> bool {
    let proximity_to_ideal = mnist_distance(test_tensor, ideal_tensor);
    let proximity_to_threshold = mnist_distance(test_tensor, threshhold_tensor);

    proximity_to_ideal < proximity_to_threshold
}

fn main() -> anyhow::Result<()> {
    // Get the images from the directories and convert them to tensors
    let threes_tensor_list = transform_dir_into_tensors("dataset/training/3");
    let sevens_tensor_list = transform_dir_into_tensors("dataset/training/7");

    // Stack tensors on one axis
    let stacked_three_tensor = stack_tensors_on_axis(&threes_tensor_list, 0);
    let stacked_seven_tensor = stack_tensors_on_axis(&sevens_tensor_list, 0);

    // Calculate the ideal shape of the digit by averaging the stack of all the digits on the first axis
    let ideal_three = average_tensor_on_axis(&stacked_three_tensor, 0);
    let ideal_seven = average_tensor_on_axis(&stacked_seven_tensor, 0);

    // Get the first sample from each directory as a tensor
    let sample_three = &stacked_three_tensor.get_on_dim(0, 10).unwrap();
    let sample_seven = &stacked_seven_tensor.get_on_dim(0, 3).unwrap();

    // Calculate the loss between the ideal and the sample
    let three_mse_loss = mse(&sample_three, &ideal_three).unwrap();
    let seven_mse_loss = mse(&sample_seven, &ideal_seven).unwrap();
    dbg!(three_mse_loss, seven_mse_loss);

    let three_mnist_distance = mnist_distance(&sample_three, &ideal_three);
    let seven_mnist_distance = mnist_distance(&sample_three, &ideal_seven);
    dbg!(three_mnist_distance, seven_mnist_distance);

    // Prepare validation set
    let valid_threes_tensor_list = transform_dir_into_tensors("dataset/validation/groups/3");
    let valid_sevens_tensor_list = transform_dir_into_tensors("dataset/validation/groups/7");
    let valid_stacked_three_tensor = stack_tensors_on_axis(&valid_threes_tensor_list, 0);
    let valid_stacked_seven_tensor = stack_tensors_on_axis(&valid_sevens_tensor_list, 0);

    // Calculate the loss on the validation set
    let valid_three_mnist_distance = mnist_distance(&valid_stacked_three_tensor, &ideal_three);
    let valid_seven_mnist_distance = mnist_distance(&valid_stacked_seven_tensor, &ideal_seven);
    dbg!(valid_three_mnist_distance, valid_seven_mnist_distance);

    // Calculate the accuracy on the validation set
    let diff = is_a_three(sample_three, &ideal_three, &ideal_seven);
    dbg!(diff);
    Ok(())
}
