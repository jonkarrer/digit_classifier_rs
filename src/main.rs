mod comparing_raw_loss;
mod stochastic_gradient_decent;
pub use stochastic_gradient_decent::run_sgd;
mod tensor_tools;

fn main() -> anyhow::Result<()> {
    run_sgd()?;
    Ok(())
}
