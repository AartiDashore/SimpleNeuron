# SimpleNeuron

## Single Neuron Visualization

This repository contains a Python script to demonstrate and visualize the behavior of a single neuron or single-layer neural network (perceptron) to solve a binary classification problem (the XOR problem) using TensorFlow and Matplotlib. The script trains a simple neural network model with one neuron and plots its decision boundary.

## Overview

- **Objective**: To visualize the decision boundary of a single neuron trained on the XOR problem.
- **Libraries Used**: TensorFlow, Keras (part of TensorFlow), Matplotlib, NumPy.

## Requirements

To run the code, you'll need to install the following Python libraries:

- TensorFlow
- Matplotlib
- NumPy

You can install these libraries using pip:

```bash
pip install tensorflow matplotlib numpy
```

## Code Explanation

### Generate Synthetic Data

The synthetic data used for this demonstration represents the XOR problem:

- **Input Features (X)**:
  `X` represents the input features. In this case, we use four input vectors (binary combinations).
  - `[[0, 0], [0, 1], [1, 0], [1, 1]]`
- **Labels (y)**:
  `y` represents the output labels. For XOR, the output is `0` if both inputs are the same and `1` otherwise.
  - `[0, 1, 1, 0]` (XOR output)

### Model Definition

A simple neural network model is defined with the following characteristics:

- **Type**: Sequential model - Sequential is a linear stack of layers.
- **Layers**: `Dense(1, input_dim=2, activation='sigmoid')` creates a dense layer with one neuron and uses the sigmoid activation function, suitable for binary classification.

### Model Compilation and Training

- **Optimizer**: `SGD(learning_rate=0.1)` specifies the Stochastic Gradient Descent optimizer with a learning rate of 0.1.
- **Loss Function**: `binary_crossentropy` is used as the loss function for binary classification.

### Model Training

- **Training**: `model.fit()` trains the model for 1000 epochs with the provided (XOR) data.

### Visualization - Evaluate and Predict

A function `plot_decision_boundary` is defined to visualize the decision boundary of the trained model:

- **Grid Creation**: Generates a mesh grid covering the input feature space.
- **Prediction**: Uses the trained model to predict probabilities over the grid.
  - `model.evaluate()` computes the loss and accuracy on the provided data.
  - `model.predict()` makes predictions for the input data and prints them.
- **Plotting**:
  - Decision boundary is shown using `contourf`.
  - Original data points are plotted using `scatter`.

## How to Run

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/single-neuron-visualization.git
    cd single-neuron-visualization
    ```

2. Install the required libraries:

    ```bash
    pip install tensorflow matplotlib numpy
    ```

3. Run the script:

    ```bash
    python visualize_neuron.py
    ```

4. View the plot showing the decision boundary and data points. You should see the loss and accuracy of the model as well as the predictions for each input vector.

## Example Output

The script will generate a plot displaying:

- The decision boundary learned by the single neuron.
- The original data points colored by their class labels.

The decision boundary illustrates how the single neuron separates different classes in the XOR problem.

![Output plot](https://github.com/AartiDashore/SimpleNeuron/blob/main/output.png)

## Additional Notes

- Activation Function: The sigmoid activation function outputs values between 0 and 1, which is suitable for binary classification problems.
- Optimizer: SGD (Stochastic Gradient Descent) is a basic optimization algorithm. For more complex problems, consider using other optimizers like Adam.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! If you have any improvements or suggestions, please feel free to create a pull request or open an issue.
