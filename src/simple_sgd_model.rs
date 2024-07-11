#![allow(unused_variables, dead_code)]

use candle_core::{Device, Tensor, Var};
use candle_nn::{loss::mse, Optimizer, SGD};

// This is a basic example of a deep learning algorithm that uses SGD as the loss optimization method.
pub fn run_sgd() -> anyhow::Result<()> {
    let time = Tensor::arange(1.0, 20.0, &Device::Cpu)?;
    let speed = &Tensor::rand(0.0, 70.0, 19, &Device::Cpu)?;

    // 1. Initialize parameters
    let params = Var::from_tensor(&Tensor::randn(35.0, 10.0, 3, &Device::Cpu)?)?;
    dbg!(&params);

    // 2. Calculate predictions
    let predictions = quad(&time, &params)?;
    dbg!(&predictions);

    // 3. Calculate loss
    let loss = mse(&predictions, &speed)?;
    dbg!(&loss);

    // 4. Calculate gradients
    let mut optimizer = SGD::new(vec![params.clone()], 1e-5)?;
    optimizer.backward_step(&loss)?;

    // 5. Step the weights
    // 6. Repeat the above steps
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
