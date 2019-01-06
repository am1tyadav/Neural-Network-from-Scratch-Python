# Create and Train a Neural Network in Python

A simple implementation to create and train a neural network in python. At the moment, this is a fairly simple and straight-forward implementation which is meant to be used for learning about neural networks and deep learning.

Use the class `model` to instantiate a neural network. This instance can be trained using the `train` method. Currently, this model can be applied only on binary classification problems. 

## Using the Model

Use the model as following:
```
m = model()
m.add_layers([2, 5, 3, 1])
X = TRAINING EXAMPLES
Y = TRAINING LABELS

m.train(X, Y, iterations = 50, alpha = 0.1, verbose = False)
```
Keep the following points in mind:

* In the `add_layers()` method, pass on a list of nodes you want for each layer, starting with the number of features as the first value. Each layer is a densely connected layer. The output node must be set to 1 currently (since only binary classification is implemented at the moment)
* You can add as many hidden layers as you want. The list of layers should look something like this: `[NUMBER OF FEATURES, NODES OF LAYER 2, NODES OF LAYER 3...., NODES OF LAYER K, 1]`
* The shape of examples must be (NUMBER OF FEATURES, NUMBER OF EXAMPLES)
* The shape of labels must be (1, NUMBER OF EXAMPLES)

## Training Parameters

When calling the `model.train()` method, you can use the following parameters (Parameters are listed in sequence):

* `X` = training examples
* `Y` = training labels
* `iterations` = default value is set to 10. This is the number of times the training loop will run for
* `alpha` = default value is set to 0.001. This is the learning rate
* `decay` = default value is set to True. This is a boolean which will check for whether or not learning rate decays as training progresses
* `decay_iter` = default value is 5. This is the number of iterations after which learning rate will be decayed
* `decay_rate` = default valye is 0.9. This is the rate with which the learning rate will be reduced i.e. if decay was `alpha`, after number of `decay_iter` iterations, the new value will be `alpha * decay_rate` 
* `stop_decay_counter` = default value is 100. This is the maximum number of changes that can happen to the learning rate while training
* `verbose` = default is set to True. If this is set to true, the model will display the value for Cost after each iteration
* `lam` = default is set to 0. This is the l2 regularization parameter

## Predictions, Evaluations and Plots

You can use the `model.evaluate(X, Y)` method to evaluate your test set(s). This returns a `float` accuracy score. You can use the `model.predict(X)` method to get predictions on new data. This will return a numpy array with the predicted labels (0 or 1). There are three plotting methods you can use after training to plot cost, accuracy and learning rate called `model.plot_cost()`, `model.plot_acc()` and `model.plot_alpha()`

## Activation Functions

Currently, all the hidden units will have *relu* activation function and the output layer has the *sigmoid* activation function

## Training Loop

The training loop does the following:

* As a first step, it initializes all the parameters of all the layers
* Then, the loop starts for the given number of iterations (full training set is used in each iteration)
* Inside the loop, forward propagation is applied, then the cost is calculated, then the gradients are calculated and finally, the parameters are updated.

## Example - Recognising Hand Written 0s and 1s from MNIST

I wanted to apply the neural network model to the MNIST dataset, and perhaps I'll do that in the next iteration, but at the moment the model can only do binary classification, not multi-class classification. So, I decided to try and train the model on only the hand-written 0s and 1s from the MNIST dataset.

The model architecture was `[784, 64, 64, 1]` and was trained for 60 iterations (batch gradient descent) with learning rate set to 0.003 and regularization parameter set to 4. The resulting accuracy on a test set was ~98.8%.
