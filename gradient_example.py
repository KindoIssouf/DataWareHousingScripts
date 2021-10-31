import numpy

"""Gradient descent for linear regression.

We will try to learn this linear regression model:
   w_0 1 + w_1 x = y

where we want to predict the value of y, from the value of x.  We want
to learn the values of the weights w_0 and w_1 from the training data
(x,y).

"""

# we shall simulate data (x,y) with the following weights
w_0,w_1 = 5,7
N = 10 # size of the dataset

# per-instance sum-squared-error
def loss(A,y,w):
    N = len(A)      # length of dataset
    a = A@w - y     # error
    return a.T@a/N  # average squared error

# gradient of the loss
def gradient(A,y,w):
    N = len(A)            # length of dataset
    a = 2*(A@w - y).T@A/N # gradient as row vector
    return a.T            # return as column vector

def optimize(A,y,iters=10,rate=0.1):
    n = len(A.T) # number of features
    w = numpy.random.random([n,1]) # initial weights
    for i in range(iters):
        cur_loss = loss(A,y,w)        # current loss
        cur_grad = gradient(A,y,w)    # current gradient
        cur_mag = cur_grad.T@cur_grad # magnitude of gradient
        print("iter: %d loss: %.4g grad: %.4g" % (i,cur_loss,cur_mag))
        w = w - rate*cur_grad         # take a gradient step
    return w # final learned weights


if __name__ == '__main__':
    """We want to set up the linear regression problem using the linear
    system:
      Aw = y
    
    where
    
      A = [ 1; x ]  
    
    where
    
      [1; x].T  [w_0 w_1]  = w_0 1 + w_1 x
      
      
       [w_0 w_1].T [1; x]  = w_0 1 + w_1 x
        
         2*1     .  1*n     gives   2*n yeahhhh
     """

    w = numpy.matrix([w_0,w_1]).T           # weight vector
    x = numpy.random.random([N,1])          # random data x
    ones = numpy.ones([N,1])                # vector of ones
    A = numpy.concatenate([ones,x],axis=1)  # the A matrix: [1; x]
    noise = numpy.random.random([N,1])      # noise, for simulating data
    y = A@w                         # simulated (noisy) labels
   # 10*2 @ 2*1 = 10 * 1
    # learn the weights, where we specify the number of iterations and the
    # learning rate.  increase the number of iterations to get better
    # weights
    new_w = optimize(A,y,iters=5000,rate=0.01)
    print(new_w)
    # when the loss is at a minimum, the magnitude of the gradient is zero

