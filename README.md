# Create and Train a Neural Network in Python

An implementation to create and train a simple neural network in python - just to learn the basics of how neural networks work

## Usage

Create a tuple of layers where each element is a tuple as well

The first element of this tuple needs to be the actual layer, and the second element needs to be the activation function applied to the layer

```python
from nn.layer import Dense
from nn.activation import ReLU, Sigmoid

layers = (
    (Dense(64), ReLU()),
    (Dense(64), ReLU()),
    (Dense(1), Sigmoid())
)
```

The model can then be created using the NeuralNetwork class

```python
from nn.loss import BinaryCrossEntropy
from nn.model import NeuralNetwork

model = NeuralNetwork(
    layers=layers,
    loss=BinaryCrossEntropy(),
    learning_rate=1.
)
```
The model can then be trained:

```python
model.fit(x_train, y_train)
```

Please take a look at this [notebook](example.ipynb) for a detailed example

## Training Loop

The training loop is in the `fit` method in `NeuralNetwork`:

```python
class NeuralNetwork(IModel):
    ...
    
    def fit(self, examples, labels, epochs):
        self._input = examples

        for epoch in range(1, epochs + 1):
            _ = self(self._input) # [1]
            loss = self._loss(self._output, labels) # [2]
            self.backward_step(labels) # [3]
            self.update() # [4]
    ...
```
At the moment, one iteration is on the entire training set and mini-batch is not implemented. 
In each iteration, we take a forward pass through the model `self(self._input)`. 
Then loss is computed. Loss computation is only necessary if you plan to use the loss in some way - eg. log the loss. 

The backward pass `self.backward_step(labels)` goes from the output layer, all the way 
back to the inputs to compute gradients for all the learnable parameters. Once this is done, 
we can update the learnable parameters with the `self.update()` method.

### 1. Forward Pass

Forward pass is executed when the model instance is called:

```python
class NeuralNetwork(IModel):
    ...
    def __call__(self, input_tensor):
        output = input_tensor

        for layer, activation in self._layers:
            output = layer(output)
            output = activation(output)

        self._output = output
    ...
```
The tuple of layers in the `self._layers` parameter is actually a tuple of tuples where 
each tuple has a layer (e.g. Dense), and an activation (e.g. ReLU).

### 2. Compute Loss

A loss function is required when instantiating the model. The loss function must implement the `ILoss` protocol 
which returns computed loss when the loss function instance is called.

### 3. Backward Pass

Backward pass computes gradients for all learnable parameters of the model:

```python
class NeuralNetwork(IModel):
    ...
    def backward_step(self, labels: np.ndarray):
        da = self._loss.gradient(self._output, labels)

        for index in reversed(range(0, self._num_layers)):
            layer, activation = self._layers[index]

            if index == 0:
                prev_layer_output = self._input
            else:
                prev_layer, prev_activation = self._layers[index - 1]
                prev_layer_output = prev_activation(prev_layer.output)

            dz = np.multiply(da, activation.gradient(layer.output))
            layer.grad_weights = np.dot(dz, np.transpose(prev_layer_output)) / self._num_examples
            layer.grad_weights = layer.grad_weights + \
                (self._regularization_factor / self._num_examples) * layer.weights
            layer.grad_bias = np.mean(dz, axis=1, keepdims=True)
            da = np.dot(np.transpose(layer.grad_weights), dz)
    ...
```
After calculating gradients from the loss function, we iterate over the layers 
backwards all the way to the input to compute the gradients for all learnable parameters. 
The computed gradients for each layer are stored in the layer instance itself - i.e 
`layer.grad_weights` and `layer.grad_bias`.

When the loop reaches the first layer, there is no previous output to it. Therefore, we set 
`prev_layer_output` to `self._input` - i.e. the input to the model, the examples 

### 4. Update the Parameters

Finally, the learnable parameters (weights and biases) are updated:

```python
class NeuralNetwork(IModel):
    ...
    def update(self):
        for layer, _ in self._layers:
            layer.update(self._learning_rate)
    ...
```

## Next Steps

1. Separate the optimization logic from the layer and model classes
2. Learning rate scheduler callback
3. Way to implement non trainable layers like Dropout
4. Way to save and load model parameters
