# Create and Train a Neural Network in Python

A simple implementation of neural networks in python. Use the class `model` to instantiate a neural network. This instance can be trained using the `train` method. Currently, this model can be applied only on binary classification problems.

## Using the Model

Use the model as following:
`
m = model()
m.add_layers([2, 5, 3, 1])
X = TRAINING EXAMPLES
Y = TRAINING LABELS

m.train(X, Y, iterations = 50, alpha = 0.1, verbose = False)
`
Keep the following points in mind:

* In the `add_layers` method, pass on a list of nodes you want for each layer, starting with the number of features as the first value. Each layer is a densely connected layer. The output node must be set to 1 currently (since only binary classification is implemented at the moment)
* The shape of examples must be (NUMBER OF FEATURES, NUMBER OF EXAMPLES)
* The shape of labels must be (1, NUMBER OF EXAMPLES)
