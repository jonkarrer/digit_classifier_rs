use candle_core::{DType, Device, Tensor, TensorId, Var};
use candle_nn::{loss::mse, Optimizer, VarBuilder, SGD};
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

fn mnist_distance(a: &Tensor, b: &Tensor) -> Tensor {
    let result = a.broadcast_sub(b).expect("Failed to broadcast subtraction");
    let abs_result = result.abs().expect("Failed to calculate absolute value");

    let last_axis_index = result.dims().len() - 1;
    let second_last_axis_index = result.dims().len() - 2;

    abs_result
        .mean((second_last_axis_index, last_axis_index))
        .expect("Failed to calculate mean")
}

fn is_a_three(test_tensor: &Tensor, ideal_tensor: &Tensor, threshhold_tensor: &Tensor) -> Tensor {
    let proximity_to_ideal = mnist_distance(test_tensor, ideal_tensor);
    let proximity_to_threshold = mnist_distance(test_tensor, threshhold_tensor);

    proximity_to_ideal
        .lt(&proximity_to_threshold)
        .expect("Failed to compare tensors")
        .to_dtype(candle_core::DType::F32)
        .expect("Failed to convert tensor to f32")
}

fn accuracy_measure(matching_set_eval: &Tensor, mismatching_set_eval: &Tensor) -> f32 {
    let matching_set_average = matching_set_eval
        .mean_all()
        .expect("Failed to calculate mean")
        .to_vec0::<f32>()
        .expect("Failed to convert tensor to f32");

    let mismatching_set_average = 1.0
        - mismatching_set_eval
            .mean_all()
            .expect("Failed to calculate mean")
            .to_vec0::<f32>()
            .expect("Failed to convert tensor to f32");

    (matching_set_average + mismatching_set_average) / 2.0
}

fn main() -> anyhow::Result<()> {
    // // Get the images from the directories and convert them to tensors
    // let threes_tensor_list = transform_dir_into_tensors("dataset/training/3");
    // let sevens_tensor_list = transform_dir_into_tensors("dataset/training/7");

    // // Stack tensors on one axis
    // let stacked_three_tensor = stack_tensors_on_axis(&threes_tensor_list, 0);
    // let stacked_seven_tensor = stack_tensors_on_axis(&sevens_tensor_list, 0);

    // // Calculate the ideal shape of the digit by averaging the stack of all the digits on the first axis
    // let ideal_three = average_tensor_on_axis(&stacked_three_tensor, 0);
    // let ideal_seven = average_tensor_on_axis(&stacked_seven_tensor, 0);

    // // Get the first sample from each directory as a tensor
    // let sample_three = &stacked_three_tensor.get_on_dim(0, 10).unwrap();
    // let sample_seven = &stacked_seven_tensor.get_on_dim(0, 3).unwrap();

    // // Calculate the loss between the ideal and the sample
    // let three_mse_loss = mse(&sample_three, &ideal_three).unwrap();
    // let seven_mse_loss = mse(&sample_seven, &ideal_seven).unwrap();
    // dbg!(three_mse_loss, seven_mse_loss);

    // let three_mnist_distance = mnist_distance(&sample_three, &ideal_three);
    // let seven_mnist_distance = mnist_distance(&sample_three, &ideal_seven);
    // dbg!(three_mnist_distance, seven_mnist_distance);

    // // Prepare validation set
    // let valid_threes_tensor_list = transform_dir_into_tensors("dataset/validation/groups/3");
    // let valid_sevens_tensor_list = transform_dir_into_tensors("dataset/validation/groups/7");
    // let valid_stacked_three_tensor = stack_tensors_on_axis(&valid_threes_tensor_list, 0);
    // let valid_stacked_seven_tensor = stack_tensors_on_axis(&valid_sevens_tensor_list, 0);

    // // Calculate the loss on the validation set
    // let valid_three_mnist_distance = mnist_distance(&valid_stacked_three_tensor, &ideal_three);
    // let valid_seven_mnist_distance = mnist_distance(&valid_stacked_seven_tensor, &ideal_seven);
    // dbg!(valid_three_mnist_distance, valid_seven_mnist_distance);

    // // Check if the sample is a three
    // let test_a_three = is_a_three(sample_three, &ideal_three, &ideal_seven);
    // dbg!(test_a_three.to_scalar::<f32>().unwrap() == 1.0);

    // // Check if the validation set is a three
    // let three_training_run = is_a_three(&valid_stacked_three_tensor, &ideal_three, &ideal_seven);
    // let seven_training_run = is_a_three(&valid_stacked_seven_tensor, &ideal_three, &ideal_seven);

    // dbg!(accuracy_measure(&three_training_run, &seven_training_run));

    // Simple gradient descent example
    // let xt = Tensor::from_vec(vec![3.0, 4.0, 6.0], 3, &candle_core::Device::Cpu)?;
    // let vars = Var::from_tensor(&xt)?;
    // dbg!(vars.powf(2.0).unwrap().backward()?);

    // SGD Example
    let time = Tensor::arange(1.0, 20.0, &Device::Cpu)?;
    let speed = &Tensor::rand(0.0, 70.0, 19, &Device::Cpu)?;

    // 1. Initialize parameters
    let params = Var::from_tensor(&Tensor::randn(35.0, 10.0, 3, &Device::Cpu)?)?;
    dbg!(&params);

    // 2. Calculate predictions
    let predictions = quad(&time, &params)?;

    // 3. Calculate loss
    let loss = mse(&predictions, &speed)?;

    // 4. Calculate gradients
    let mut optimizer = SGD::new(vec![params.clone()], 1e-5)?;
    optimizer.backward_step(&loss)?;

    // 5. Step the weights
    // 6. Repeat
    for _ in 0..10 {
        let step_result = apply_step(&params, &time, &speed)?;
        dbg!(&step_result);
    }

    // 7. Stop
    dbg!(&params);

    Ok(())
}

fn apply_step(params: &Var, time: &Tensor, speed: &Tensor) -> anyhow::Result<Tensor> {
    // 2. Calculate predictions
    let predictions = quad(&time, &params)?;

    // 3. Calculate loss
    let loss = mse(&predictions, &speed)?;

    // 4. Calculate gradients
    let mut optimizer = SGD::new(vec![params.clone()], 1e-5)?;
    optimizer.backward_step(&loss)?;

    Ok(loss)
}

fn quad(time: &Tensor, parameters: &Var) -> anyhow::Result<Tensor> {
    let a = parameters.get(0)?;
    let b = parameters.get(1)?;
    let c = parameters.get(2)?;

    let t_squared = time.powf(2.0)?;
    let term_1 = a.broadcast_mul(&t_squared)?;
    let term_2 = b.broadcast_mul(&time)?;

    let res = term_1.add(&term_2)?.broadcast_add(&c)?;

    Ok(res)
}
