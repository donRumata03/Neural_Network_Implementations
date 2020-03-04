# My Neural network implementations

This repository contains my own implementations of basic machine learning algorithms.
 Here are some of them:
 
### 1) Polynomial Regression
To watch polynomial regression\`s work you only need to run file 
    `polynomial_regression.py`
There are some points randomly generated around a curve of any pow polynom.
After that the algorithm starts its work and approximates that points by required function.
To do that gradient descent and newton method are used.

### 2) Perceptron
To watch perceptron\`s work you only need to run file
    ``perceptron.py``
There\`s a class of perceptron of custom layer geometry 
( that means, that the user can change only a few symbols to change the layer number and configuration )
To test its work there is an example:
The task, for which it\`s learned by default, is determining if the given point belongs to the circle or not.
Of course, firstly, there is a generator of such datasets, which randomly chooses some points, for example, 1000, which are situated inside the circle and the same amount, which are not.
Than, Stochastic Gradient Descent optimizes the loss function 
( personally a use the configuration {80 X 40 X 20}, which is more than enough to classify the points ideally, so, you can also use {40 X 20} of something like this for NN architecture ) 

### 3) Convolutional Neural Network (CNN)
It\`s one of the word best decisions for computer vision, face recognition and e.t.c. ...
It\`s in progress af for now... But believe me is\`s being actively developed...


### 4) Recurrent Neural Network (RNN)
This is going to be used in my robot to improvise notes according to chords recognised.
Unfortunately, it is not even being developed yet, but I hope to implement it soon on my own.
