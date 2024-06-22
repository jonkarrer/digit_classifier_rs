use std::{fs, path::Path};

use candle_core::Tensor;
use candle_nn::loss::mse;

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

fn main() -> anyhow::Result<()> {
    // Get the images from the directories and convert them to tensors
    let threes_dir: Vec<Tensor> = dir_walk("dataset/training/3")?
        .into_iter()
        .map(|path| tensor_from_image(&path).expect("Failed to convert image"))
        .collect();

    let sevens_dir: Vec<Tensor> = dir_walk("dataset/training/7")?
        .into_iter()
        .map(|path| tensor_from_image(&path).expect("Failed to convert image"))
        .collect();

    // Stack tensors on one axis
    let threes = Tensor::stack(&threes_dir, 0)?
        .to_dtype(candle_core::DType::F32)
        .unwrap();

    let sevens = Tensor::stack(&sevens_dir, 0)?
        .to_dtype(candle_core::DType::F32)
        .unwrap();

    // Calculate the ideal shape of the digit by averaging the stack of all the digits on the first axis
    let _ideal_three = threes.mean(0).unwrap();
    let ideal_seven = sevens.mean(0).unwrap();

    // Get the first sample from each directory as a tensor
    let sample_three = &threes_dir[0].to_dtype(candle_core::DType::F32).unwrap();
    let _sample_seven = &sevens_dir[0].to_dtype(candle_core::DType::F32).unwrap();

    // Calculate the loss between the ideal and the sample
    let loss = mse(&sample_three, &ideal_seven).unwrap();
    dbg!(loss);

    Ok(())
}
