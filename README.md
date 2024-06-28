# Digit Classifier

Attempt to writes a digit classifier with some bare metal tools, no pre trained models. Our goal is to create a model that can recognize handwritten digits. This follows the Practical Deep Learning for Programmers bool, chapter 4. [Here is the link to the notebook](https://colab.research.google.com/github/fastai/fastbook/blob/master/04_mnist_basics.ipynb#scrollTo=ON4EYFojQo8o).

## Installation

Install [candle-core](https://github.com/huggingface/candle)

```bash
cargo add --git https://github.com/huggingface/candle.git candle-core
```

Install [candle-nn](https://github.com/huggingface/candle)

```bash
cargo add --git https://github.com/huggingface/candle.git candle-nn
```

Install [image](https://github.com/image-rs/image)

```bash
cargo add image
```

## Dataset

The dataset used is the from Kaggle [MNIST as .jpg](https://www.kaggle.com/datasets/scolianni/mnistasjpg)

## Baseline Model

Our first goal is to train a simple model that can recognize 3s and 7s. To do that we first need a baseline, and so we need to break this down with some questions and brainstorm some ideas.

1. How does a computer understand images?
   - A grid of width and height containing all the pixels of an image with RGB values.
2. How might a computer recognize the differences?
   - Grouping the pixels into regions that are similar to each other. The concentration of colors in the same region will tell us something about the image.
3. What can we use the differences to learn?
   - We can create an "ideal" 3 and 7 by averaging the pixels of all the 3's and 7's respectively.
4. How can we measure the differences?
   - We can use a loss function to measure the differences.
5. How can we measure the accuracy of our model?
   - We can use a loss function to measure the accuracy of our model over the validation set.

### Creating Ideal Images

We need to create the ideal digits by stacking the images together and averaging the pixel values at each coordinate. Using Tensors and some image manipulation we can do this.

First we need to get the image paths from our dataset and put into a vector of strings.

```rust
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
// ["dataset/training/3/03.jpg", "dataset/training/3/02.jpg", "dataset/training/3/07.jpg"]
```

Next we need a function to load an image into a tensor.

```rust
fn tensor_from_image(img_path: &str) -> anyhow::Result<Tensor> {
    let img = image::open(img_path)?;
    let img_ten = candle::Tensor::from_raw_buffer(
        img.as_bytes(),
        candle_core::DType::U8,
        &[28, 28],
        &candle_core::Device::Cpu,
    )?;
    Ok(img_ten)
}
```

Combine these together to create a vector of tensors.

```rust
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
```

Now we have a vector of tensors, but what we want is a stacked tensor on the axis 0. This will allow use to do the averaging.

```rust
fn stack_tensors_on_axis(tensor_list: &[Tensor], axis: usize) -> Tensor {
    candle::Tensor::stack(tensor_list, axis)
        .expect("Failed to stack tensors on dimension")
        .to_dtype(candle_core::DType::F32)
        .expect("Failed to convert tensor to f32")
}
```

The shape of this tensor should be Tensor[number of images, 28, 28]

```rust
dbg!(stacked_tensor.shape());
// [823, 28, 28]
```

Using a built in method from Candle that is similar to pytorch, we can average the pixels on the 0 axis.

```rust
let averaged_tensor = stacked_tensor.mean(0)?;
dbg!(averaged_tensor.shape());
// [28, 28]
```

This will render a very blurry image of the "ideal" digit. What happened is that on the 0 axis, the pixel values were averaged across all x and y coordinates. This is why the image is blurry. It is the average of the entire directory of images.

### Measuring Distance

Now we need to measure the distance between the ideal and the actual test image. There are two basic loss functions for this.

- L1 Norm : The mean of the absolute values of the differences. MAD
- L2 Norm : The square root of the sum of the squared differences. MSE

Thankfully, many ML Libraries have built in functions for these loss functions. Let's use MSE from candle's neural network library.

```rust
use candle_nn::loss::mse;

let ideal_three = average_tensor_on_axis(&stacked_three_tensor, 0);
let sample_three = &stacked_three_tensor.get_on_dim(0, 10).unwrap(); // Pluck the 10th image from the tensor

let three_mse_loss = mse(&sample_three, &ideal_three).unwrap();
dbg!(three_mse_loss.shape());
// Tensor[34.235]
```

The problem is that this tells us little about little. We have an arbitrary number that is the difference between the ideal and the actual image. But how do we know if this is a 3 or not?

We can compare it to our ideal 7 image, and this will give use more information. At least we can tell if it is closer to a 7 than a 3. Since were are manually pulling the 3 and 7 directory, we have full control of the comparison and expectations. This won't be the case later, but for now that's ok.

First, let's create a distance function that can leverage Broadcasting. When we go to test our validation set later, Broadcasting will help up grow our tensors to the same shape.

```rust
fn mnist_distance(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // If a and b are not the same shape, broadcast them to the same shape and then subtract them
    // Example: a: Tensor[3478, 28, 28] - b: Tensor[28, 28] = Tensor([0.1050, 0.1526, 0.1186,  ..., 0.1122, 0.1170, 0.1086])
    // The resulting tensor is a vector of all the differences between images in a compared to the single tensor in b
    let result = a.broadcast_sub(b)?;

    // Next, we want the absolute value of all the differences
    let abs_result = result.abs()?;

    // We want to take the mean ranging over the values indexed by the last two axes of the tensor.
    // The last two axes are the horizontal and vertical dimensions of an image
    let last_axis_index = result.dims().len() - 1;
    let second_last_axis_index = result.dims().len() - 2;
    abs_result.mean((second_last_axis_index, last_axis_index))
}
```

We have more flexibility in the distance function. Now we can leverage this for a single test, one that compares the test image to an ideal 3 and an ideal 7.

```rust
fn is_a_three(test_tensor: &Tensor, ideal_tensor: &Tensor, threshhold_tensor: &Tensor) -> Tensor {
    let proximity_to_ideal = mnist_distance(test_tensor, ideal_tensor); // Compare against the ideal 3
    let proximity_to_threshold = mnist_distance(test_tensor, threshhold_tensor); // Compare against the ideal 7

    // Compare proximity to ideal and proximity to threshold
    proximity_to_ideal
        .lt(&proximity_to_threshold)
        .expect("Failed to compare tensors")
        .to_dtype(candle_core::DType::F32)
        .expect("Failed to convert tensor to f32")
}

main() {
    let ideal_three = average_tensor_on_axis(&stacked_three_tensor, 0);
    let ideal_seven = average_tensor_on_axis(&stacked_seven_tensor, 0);
    let sample_three = &stacked_three_tensor.get_on_dim(0, 10).unwrap();

    let test_a_three = is_a_three(sample_three, &ideal_three, &ideal_seven);
    dbg!(test_a_three.to_scalar::<f32>().unwrap() == 1.0);
    // true
}
```

Because we have control over our inputs and expectations, meaning we know we are passing a three in, and we know that it is compared to a vastly different digit 7. So we can trust that our model is able to predict a 3.

### Validation set testing

Our baseline model needs an accuracy measure for us to evaluate it. This is the final step of our goal. We can leverage the `is_a_three` function in the validation set to test our model. Luckily for for use, we have already baked in the broadcasting functionality. So we can leverage that in the validation set without much more code.

First we prepare our validation set similar to earlier.

```rust
// Prepare validation set
let valid_threes_tensor_list = transform_dir_into_tensors("dataset/validation/groups/3");
let valid_sevens_tensor_list = transform_dir_into_tensors("dataset/validation/groups/7");
let valid_stacked_three_tensor = stack_tensors_on_axis(&valid_threes_tensor_list, 0);
let valid_stacked_seven_tensor = stack_tensors_on_axis(&valid_sevens_tensor_list, 0);
```

Now we can pass in the validation set Tensor into our distance function that will compare it against the ideal digit with broadcasting. This will return a vector of the result of each less than comparison done in the `is_a_three` function as a float.

```rust
let three_training_run = is_a_three(&valid_stacked_three_tensor, &ideal_three, &ideal_seven);
let seven_training_run = is_a_three(&valid_stacked_seven_tensor, &ideal_three, &ideal_seven);
dbg!(three_training_run);
// Tensor[1.0, 0.0, 1.0, 1.0, 0.0,...]
```

We can take this comparison result Tensor and use it to measure the accuracy of our model.

```rust
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

main() {
    dbg!(accuracy_measure(&three_training_run, &seven_training_run));
    // 96.66667
}
```

### Baseline Model Conclusion

While impressive we have achieved a 96% accuracy on our baseline model, the machine did not actually learn anything yet. We have simply stacked the cards in our favor, and with some creative math, forced the predictions in our favor. Next, we will leverage Stochastic Gradient Descent to create a better model.
