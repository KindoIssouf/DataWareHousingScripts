# CS4412 : Data Mining
# Fall 2021
# Kennesaw State University

"""In this script, we use tensorflow to model a Boolean function as a
logistic regression model.  We learn this model using TensorFlow,
which performs gradient descent automatically, from the definition of
the model (i.e., it computes the derivatives automatically).

Try these things:
1) try to learn a Node model of the OR function
2) try to learn a Node model of the AND function
   (update the labels to reflect an AND function)
3) try to learn a Node model of the XOR function
   (update the labels to reflect an XOR function)
   (this will fail)
4) try to learn a Circuit model of the XOR function
   (use the Circuit() constructor instead of the Node() constructor)
   (this will probably fail)
5) try to learn a Circuit model of the XOR function
   (this time, use the relu activation in the Node model)
   (this will probably work, possibly after a few tries)
"""

import tensorflow as tf
import numpy as np

# Here, we enumerate all 4 possible inputs of a Boolean function over
# two variables.  We use this as a dataset.
data = np.array([[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]], dtype="float32")
# Initially, the desired outputs here form an "OR" function.  Later,
# we can change the labels to represent an "AND" function, or an "XOR"
# function.
labels = np.array([[0],
                   [1],
                   [1],
                   [1]], dtype="float32")


# This is our own implementation of the mean-squared-error
def my_loss(x, y):
    a = x - y
    return a * a


# using Keras, the following class models a 2-variable logistic
# regression model of the following form:
#   sigmoid( w_1 x_1 + w_2 x_2 + b )
#
# where sigmoid() is the sigmoid (or logistic) function:
#   sigmoid(x) = (1+exp(-x))^(-1)
class Node(tf.keras.Model):
    # we first specify the weights and bias
    # which are the parameters of the model, that we will learn
    def __init__(self):
        super(Node, self).__init__()
        seed = tf.random.uniform([2, 1])
        self.w = tf.Variable(seed)  # weights
        seed = tf.zeros([1, 1])
        self.b = tf.Variable(seed)  # bias

    # this implements a model, i.e., for a given input x, we compute
    # the value of the output y
    def call(self, x):
        # the sigmoid function works well for AND/OR functions, but
        # not when we try to learn the XOR
        return tf.sigmoid(x @ self.w + self.b)
        # the ReLU works better than sigmoid in practice
        # return tf.nn.relu( x@self.w + self.b )


# in theory, the Node class can represent any AND, OR or NOT function.
# We can compose these models into a 3-node circuit, which should in
# principle be able to represent an XOR function.
class Circuit(tf.keras.Model):
    # the parameters of the model are the parameters of each Node
    def __init__(self):
        super(Circuit, self).__init__()
        self.n1 = Node()
        self.n2 = Node()
        self.n3 = Node()

    # this implements a model, i.e., for a given input x, we compute
    # the value of the output y
    def call(self, x):
        a = self.n1(x)
        b = self.n2(x)
        c = self.n3(tf.concat([a, b], axis=1))
        return c


if __name__ == '__main__':
    # initialize our model (either a Node or a Circuit)
    model = Node()
    # model = Circuit()

    # "compile" our model ... here, we simply specify the gradient descent
    # algorithm (the adam optimizer), the loss function, and the metric
    # used to measure how well the model is doing
    model.compile(optimizer="adam",
                  # loss=tf.keras.losses.MeanSquaredError(),
                  loss=my_loss,
                  metrics=[tf.keras.metrics.BinaryAccuracy()])

    # train the model from the data/labels, using epochs as the number of
    # iterations (to successfuly train the model, we may need to run
    # with a large enough number of epochs)
    model.fit(data, labels, epochs=10000)

    print("= the labels that we want to learn:")
    print(labels)
    print()
    print("= the labels that we actually learned:")
    print(model.predict(data))
