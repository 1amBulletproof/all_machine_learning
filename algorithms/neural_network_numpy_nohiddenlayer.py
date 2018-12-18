#@src: http://skipperkongen.dk/2018/02/24/how-to-do-backpropagation-in-numpy/
import numpy as np

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)
# make printed output easier to read
# fewer decimals and no scientific notation
np.set_printoptions(precision=3, suppress=True)

# learning rate
lr = 1e-2

# sigmoid function
def sigmoid(x,deriv=False):
    if deriv:
        result = x*(1-x)
    else:
        result = 1/(1+np.exp(-x))
    return result

# leaky ReLU function
def prelu(x, deriv=False):
    c = np.zeros_like(x)
    slope = 1e-1
    if deriv:
        c[x<=0] = slope
        c[x>0] = 1
    else:
        c[x>0] = x[x>0]
        c[x<=0] = slope*x[x<=0]
    return c

# non-linearity (activation function)
nonlin = prelu # instead of sigmoid

# initialize weights randomly with mean 0
W = 2*np.random.random((3,1)) - 1

# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
# output dataset
y = np.array([[0,0,1,1]]).T

print('X:\n', X)
print('Y:\n', y)
print()

for iter in range(1000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,W))

    # how much did we miss?
    l1_error = y - l1

    # compute gradient (slope of activation function at the values in l1)
    l1_gradient = nonlin(l1, True)
    # set delta to product of error, gradient and learning rate
    l1_delta = l1_error * l1_gradient * lr

    # update weights
    W += np.dot(l0.T,l1_delta)

    if iter % 100 == 0:
        print('pred:', l1.squeeze(), 'mse:', (l1_error**2).mean())

print ("Output After Training:")
print ('l1:', np.around(l1))
