# Create and Train a Neural Network in Python

An implementation to create and train a simple neural network in python - just to learn the basics of how neural networks work

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

The model can then be created using the MLP class

```python
from nn.loss import BinaryCrossEntropy
from nn.model import MLP

model = MLP(
    layers=layers,
    loss=BinaryCrossEntropy(),
    lr=1.
)
```
The model can then be trained:

```python
model.fit(x_train, y_train)
```

Please take a look at this [notebook](example.ipynb) for a detailed example

