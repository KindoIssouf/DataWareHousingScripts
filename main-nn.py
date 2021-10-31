# CS4412 : Data Mining
# Fall 2021
# Kennesaw State University

"""In this script, we use TensorFlow to manually model a feedforward
neural network, neuron-by-neuron and layer-by-layer.  We do this to be
more familiar with how neural networks work; it is much easier and
efficient to use TensorFlow abstractions to define a neural network.

We train this neural network on the mnist dataset, a dataset of
handwritten images of digits.

Things to try:

1) try using the sigmoid activation function instead of relu (training
will probably fail) 

2) try increasing the size of the hidden layer, until the neural
network starts to overfit.

3) try to use the SlowLayer's instead of the Layer's in the
initializer of the NeuralNetwork class, and see how much slower it
becomes.  Using a matrix-multiply to evaluate all neurons at once,
rather than one-by-one is much faster.  If you have an nvidia GPU, it
might be MUCH faster to use Layer instead of SlowLayer (since GPUs are
really good at doing matrix multiply's really fast).

4) if you want to try a different dataset, try Fashion-MNIST, which
are 28x28 images of different types of clothes.  (the same dimension
as MNIST)

"""

import tensorflow as tf
import pickle

with open("mnist/mnist-train-images",'rb') as f: train_images = pickle.load(f)
with open("mnist/mnist-train-labels",'rb') as f: train_labels = pickle.load(f)
with open("mnist/mnist-test-images",'rb') as f: test_images = pickle.load(f)
with open("mnist/mnist-test-labels",'rb') as f: test_labels = pickle.load(f)
# do one-hot-encoding of the labels
train_labels = tf.keras.utils.to_categorical(train_labels,num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels,num_classes=10)

"""A neuron is of the form:
  f( sum_i w_i x_i + b )
where x_i are our inputs
      w_i is a weight, one per input
      b is a bias term
and f is an activation function.

"""
class Neuron(tf.keras.models.Model):
    def __init__(self,input_dim,activation):
        super(Neuron,self).__init__()
        self.w = tf.random.uniform([input_dim,1]) # random weights
        self.b = tf.zeros([1,1])                  # a zero bias
        self.w = tf.Variable(self.w)
        self.b = tf.Variable(self.b)
        self.activation = activation # activation function

    # returns the output of the neuron
    def call(self,x):
        return self.activation(x@self.w + self.b)

"""A layer is a row of multiple neurons.  In this class, we represent
a layer as an explicit list of neurons.  It is faster to not represent
the neurons explicitly, and just treat each layer as a
matrix-multiply, which we do in the Layer class.

"""        
class SlowLayer(tf.keras.models.Model):
    def __init__(self,input_dim,output_dim,activation):
        super(SlowLayer,self).__init__()
        self.units = [ Neuron(input_dim,activation) \
                       for _ in range(output_dim) ]

    # returns the output of the layer
    def call(self,x):
        output = [ neuron(x) for neuron in self.units ]
        return tf.concat(output,axis=1)

"""We represent a layer of neurons as a matrix multiply.  A matrix
multiply is equal to doing a dot product of n weight vectors (each
representing a neuron), with the same input.  The bias term is now a
vector of biases.

"""
class Layer(tf.keras.models.Model):
    def __init__(self,input_dim,output_dim,activation):
        super(Layer,self).__init__()
        self.A = tf.random.uniform([input_dim,output_dim]) # random weights
        self.b = tf.zeros([1,output_dim])                  # zero biases
        self.A = tf.Variable(self.A)
        self.b = tf.Variable(self.b)
        self.activation = activation # activation function

    # returns the output of the layer
    def call(self,x):
        return self.activation( x@self.A + self.b )

"""A feedforward neural network is a sequence of layers.

"""
class NeuralNetwork(tf.keras.models.Model):
    def __init__(self,dimensions,activation):
        super(NeuralNetwork,self).__init__()
        self._layers = []
        # the input of a layer is the output of the previous layer
        last_dim = dimensions[0]   # the dimension of the last layer
        for dim in dimensions[1:]: # the dimension of the cur layer
            layer = Layer(last_dim,dim,activation)
            self._layers.append(layer)
            last_dim = dim # output of cur is input of next layer

    # returns the output of the layer
    def call(self,x):
        for layer in self._layers:
            x = layer(x)
        return x

# specify the function to use to test accuracy
accuracy_function = tf.keras.metrics.CategoricalAccuracy()

# dimensions of the neural network.  first dimension is the size of
# the input layer (28*28) and the last dimension is the size of the
# output layer (10, one for each digit)
dimensions = [ 28*28, 40, 10 ]

# construct the neural network and set it up for training
model = NeuralNetwork(dimensions,tf.nn.relu)
model.compile(optimizer="adam",
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=[accuracy_function])
# train the model
model.fit(train_images,train_labels,epochs=10)

# evaluate the model on the training data
train_predictions = model.predict(train_images)
train_acc = accuracy_function(train_predictions,train_labels)
print("train accuracy: %.2f%%" % (100*train_acc))

# evaluate the model on the testing data
test_predictions = model.predict(test_images)
test_acc = accuracy_function(test_predictions,test_labels)
print("test accuracy: %.2f%%" % (100*test_acc))

# To repeat: this is a really inefficient way of implementing a neural
# network in tensorflow.  We are just doing it like this to know what
# a neural network actually looks like.
