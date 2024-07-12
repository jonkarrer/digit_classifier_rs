mod comparing_raw_loss;
mod double_layer_linear;
mod simple_linear_model;
mod simple_sgd_model;
mod tensor_tools;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, ops::sigmoid, Linear, Module, Optimizer, VarBuilder, VarMap};
use tensor_tools::{stack_tensors_on_axis, transform_dir_into_tensors};

const IMG_DIM: usize = 28 * 28;
const RESULTS: usize = 1;
const LAYER_ONE_OUT_SIZE: usize = 100;
const LAYER_TWO_OUT_SIZE: usize = 50;
const LEARNING_RATE: f64 = 0.03;
const EPOCHS: usize = 10;

#[derive(Clone)]
struct Dataset {
    training_inputs: Tensor,
    training_labels: Tensor,
    test_inputs: Tensor,
    test_labels: Tensor,
}

struct Model {
    layer_one: Linear,
    layer_two: Linear,
    layer_three: Linear,
}

impl Model {
    fn new(vs: VarBuilder) -> Result<Self> {
        let layer_one = linear(IMG_DIM, LAYER_ONE_OUT_SIZE, vs.pp("ln1"))?;
        let layer_two = linear(LAYER_ONE_OUT_SIZE, LAYER_TWO_OUT_SIZE, vs.pp("ln2"))?;
        let layer_three = linear(LAYER_TWO_OUT_SIZE, RESULTS, vs.pp("ln3"))?;

        Ok(Self {
            layer_one,
            layer_two,
            layer_three,
        })
    }

    fn forward(&self, images: &Tensor) -> Result<Tensor> {
        let x = self.layer_one.forward(images)?;
        let x = x.relu()?;
        let x = self.layer_two.forward(&x)?;
        let x = x.relu()?;
        self.layer_three.forward(&x)
    }
}

fn train(m: Dataset, dev: &Device) -> anyhow::Result<Model> {
    // ** 1. Initialize parameters (weights and biases)
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = Model::new(vs.clone())?;
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;

    let training_inputs = m.training_inputs.to_device(dev)?;
    let training_labels = m.training_labels.to_device(dev)?;
    let test_inputs = m.test_inputs.to_device(dev)?;
    let test_labels = m.test_labels.to_device(dev)?;

    let mut final_accuracy = 0.0;

    for epoch in 1..EPOCHS + 1 {
        // ** 2. Calculate predictions.
        let logits = model.forward(&training_inputs)?;

        // ** 3. Calculate loss
        let loss = loss_with_sigmoid(&logits, &training_labels)?;

        // ** 4. Calculate gradients and step the loss
        sgd.backward_step(&loss)?;

        // ** 5. Test accuracy
        let test_logits = model.forward(&test_inputs)?;
        let test_probs = sigmoid(&test_logits)?;
        let test_preds = test_probs.ge(0.5)?;
        let test_labels_u8 = test_labels.to_dtype(DType::U8)?;

        let correct_preds = test_preds.eq(&test_labels_u8)?;
        let sum_ok = correct_preds.sum_all()?.to_dtype(DType::F32)?;
        let test_accuracy = sum_ok.to_scalar::<f32>()? / test_labels.dims()[0] as f32;
        final_accuracy = 100. * test_accuracy;
        println!(
            "Epoch: {epoch:3} Test loss: {:5.2}  Test accuracy: {:5.2}%",
            loss.to_vec1::<f32>()?[0],
            final_accuracy
        );

        // ** 6. Keep looping and then
        // ** 7. Early stopping
        if final_accuracy == 100.0 {
            break;
        }
    }

    if final_accuracy < 90.0 {
        Err(anyhow::Error::msg("The model is not trained well enough."))
    } else {
        Ok(model)
    }
}

fn main() -> anyhow::Result<()> {
    let device = Device::cuda_if_available(0)?;

    // ** 0. Prepare dataset
    let training_threes = transform_dir_into_tensors("dataset/training/3");
    let training_sevens = transform_dir_into_tensors("dataset/training/7");
    let training_threes = stack_tensors_on_axis(&training_threes, 0);
    let training_sevens = stack_tensors_on_axis(&training_sevens, 0);
    let training_inputs = Tensor::cat(&[&training_threes, &training_sevens], 0)?;
    let training_inputs = training_inputs.reshape(((), 28 * 28))?;

    let mut training_labels: Vec<f32> = vec![1f32; training_threes.dims()[0]];
    training_labels.extend(vec![0f32; training_sevens.dims()[0]]);
    let dim_length = training_labels.len();
    let training_labels = Tensor::from_vec(training_labels, (dim_length, 1), &device)?;

    let test_threes = transform_dir_into_tensors("dataset/validation/groups/3");
    let test_sevens = transform_dir_into_tensors("dataset/validation/groups/7");
    let test_threes = stack_tensors_on_axis(&test_threes, 0);
    let test_sevens = stack_tensors_on_axis(&test_sevens, 0);
    let test_inputs = Tensor::cat(&[&test_threes, &test_sevens], 0)?;
    let test_inputs = test_inputs.reshape(((), 28 * 28))?;

    let mut test_labels: Vec<f32> = vec![1f32; test_threes.dims()[0]];
    test_labels.extend(vec![0f32; test_sevens.dims()[0]]);
    let dim_length = test_labels.len();
    let test_labels = Tensor::from_vec(test_labels, (dim_length, 1), &device)?;

    let dataset = Dataset {
        training_inputs,
        training_labels,
        test_inputs,
        test_labels,
    };

    let trained_model: Model;
    loop {
        println!("Trying to train neural network.");
        match train(dataset.clone(), &device) {
            Ok(model) => {
                trained_model = model;
                break;
            }
            Err(e) => {
                println!("Error: {}", e);
                continue;
            }
        }
    }

    let test_three = run_a_final_test(&trained_model, &test_threes)?;
    println!("Test 3: Model predicts {}", test_three);

    let test_seven = run_a_final_test(&trained_model, &test_sevens)?;
    println!("Test 7: Model predicts {}", test_seven);

    Ok(())
}

fn run_a_final_test(trained_model: &Model, test_image: &Tensor) -> Result<String> {
    let final_result = trained_model.forward(
        &test_image
            .reshape(((), 28 * 28))?
            .get(0)?
            .unsqueeze(1)?
            .transpose(0, 1)?,
    )?;

    let test_probs = sigmoid(&final_result)?;
    let test_preds = test_probs.ge(0.5)?;

    let res = test_preds.get_on_dim(0, 0)?.to_vec1::<u8>()?[0];

    if res == 1 {
        Ok("Three".to_string())
    } else {
        Ok("Seven".to_string())
    }
}

fn loss_with_sigmoid(prediction: &Tensor, target: &Tensor) -> Result<Tensor> {
    let prediction = sigmoid(prediction)?;

    target
        .eq(1f32)?
        .where_cond(&Tensor::ones_like(&target)?.sub(&prediction)?, &prediction)?
        .mean(0)
}
