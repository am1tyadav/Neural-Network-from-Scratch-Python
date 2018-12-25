# Create and Train a Neural Network in Python

A simple implementation of neural networks in python. Use the class `model` to instantiate a neural network. This instance can be trained using the `train` method. Currently, this model can be applied only on binary classification problems. This is a pretty simple implementation which is meant to be used for learning about neural networks and deep learning.

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

You can use the `model.evaluate(X, Y)` method to evaluate your test set(s). This returns a `float` accuracy score. You can use the `model.predict(X)` method to get predictions on new data. This will return a numpy array with the predicted labels (0 or 1). There are three plotting methods you can use after training to plot cost, accuracy and learning rate called `method.plot_cost()`, `method.plot_acc()` and `method.plot_alpha()`

## Activation Functions

Currently, all the hidden units will have *relu* activation function and the output layer has the *sigmoid* activation function
