# CS4412 : Data Mining
# Fall 2021
# Kennesaw State University

"""This script shows how one can implement linear regression using
numpy.  We use polynomial fitting as an example."""

import numpy
from matplotlib import pylab

if __name__ == '__main__':

    # fitting a second-order polynomial (quadratic)
    # ax^2 + bx + c = y
    a,b,c = 7,-7,5                              # co-effecients
    n = 100                                     # number of data points

    # simulate some random data (x,y)
    x = numpy.random.random([n,1])              # generate features x
    noise = 0.5*numpy.random.random([n,1])      # noise
    y = a*x*x + b*x + c + noise                 # generate target values y

    # visualize the data
    pylab.figure()
    pylab.plot(x,y,'kx')

    # construct the A matrix for the system Ax=b
    ones = numpy.ones([n,1])                    # the last column of ones
    A = numpy.concatenate([x*x,x,ones],axis=1)  # [ x^2 x 1 ]
    '''
    this is like ax^2 + bx  + c = y
    the unknowns here are a,b : the weights
    if we have lots of data:
    we have x1, x2 ... xn
    therefore: 
   
    | x1^2   x1   1 |       | a |       | y1 |
    | x2^2   x2   1 | 
    | .       .   . |  *    | b |   =   | y2 |
    | .       .   . | 
    | x1^2   x1   1 |       | c |       | y3 |
    
        the shapes of there matries are
    
        n * 3        *       3*1     =     n*1   
        
        let represent them by: 
        
        A         *         x        =     y
        
    A is not a square matrix therefore can't just use the inverse to send A to the y side of the equ
    we multiply is A by the A.T, which a : A.T*A => 3*n * n*3 = 3*3 which is a square matrix :)

    finally: 
     "solve" for x:
     A^TA A x = A^T y
     (A^T A)^-1 (A^T A) x = (A^T A)^-1 A^T y
      x = (A^T A)^ - 1 A^T y    

    '''
    # "solve" for x:
    # A^TA A x = A^T y
    # (A^T A)^-1 (A^T A) x = (A^T A)^-1 A^T y
    # x = (A^T A)^ - 1 A^T y
    solution = numpy.linalg.inv(A.T@A)@A.T@y    # @ is matrix-multiply

    # visualize the fit
    _a,_b,_c = solution                         # learned co-efficients
    _x = numpy.linspace(0,1,100)                # numbers from 0 to 1
    _y = _a*_x*_x + _b*_x + _c                  # predicted y values
    pylab.plot(_x,_y,'r-')
    pylab.show()
