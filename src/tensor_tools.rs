use candle_core::Tensor;
use std::{fs, path::Path};

pub fn dir_walk(dir_path: &str) -> anyhow::Result<Vec<String>> {
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

pub fn tensor_from_image(img_path: &str) -> anyhow::Result<Tensor> {
    let img = image::open(img_path)?;
    let img_ten = Tensor::from_raw_buffer(
        img.as_bytes(),
        candle_core::DType::U8,
        &[28, 28],
        &candle_core::Device::Cpu,
    )?;
    Ok(img_ten)
}

pub fn transform_dir_into_tensors(dir_path: &str) -> Vec<Tensor> {
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

pub fn stack_tensors_on_axis(tensor_list: &[Tensor], axis: usize) -> Tensor {
    Tensor::stack(tensor_list, axis)
        .expect("Failed to stack tensors on dimension")
        .to_dtype(candle_core::DType::F32)
        .expect("Failed to convert tensor to f32")
}

pub fn average_tensor_on_axis(tensor: &Tensor, axis: usize) -> Tensor {
    tensor.mean(axis).expect("Failed to calculate mean")
}
