import numpy as np
import math

#the loss of each class given input x, correct class label index y, prediction weights W
#for the multi class SVM loss function
def SVM_L_i_vectorized(x,y,W):
	scores = W.dot(x) #predictions for each image
	margins = np.maximum(0,scores - scores[y]+1) #math for hinge loss
	margins[y] = 0 #ignore the correct class labels
	loss_i = np.sum(margins) #add all the losses for the incorrect classes
	return loss_i


#used more widely, this performs softmax loss on input x, correct class label index y,
#prediction weights W
def Softmax_L_vectorized(x,y,W):
    scores = W.dot(x) #classic example of a score function
    scores -= np.max(scores) #to avoid blowup due to log of a negative
    scores = np.exp(scores) #exponentiate each term
    scores = scores/np.sum(scores) #normalize each term
    L = -np.log(scores) #take the log
    return L[y]

def test_loss():
    x = np.random.randn(5,1) #5 total inputs
    W = np.random.randn(3,5) #output is W.dot(x) -> 3 x 1 matrix
    y = 2 #index to query the loss of
    print("x: ",x)
    print("W: ",W)
    print("y: ",y)
    print("SVM loss: ",SVM_L_i_vectorized(x,y,W))
    print("Softmax loss: ",Softmax_L_vectorized(x,y,W)[0])

test_loss()

def Loss(x,y,W):
    totalLoss = 0
    for i in range(len(y)):
        totalLoss += Softmax_L_vectorized(x,i,W)
    return totalLoss

#optimization

#very bad idea, random search for gradient descent
def randSearch():
    x_train = np.random.randn(10,1)
    y_train = np.random.randn(4,1)

    bestloss = float("inf")
    for num in range(1000):
        W = np.random.randn(10,3073)*0.0001 #random direction --> BAD
        loss = Loss(x_train,y_train,W)
        if loss < bestloss:
            bestloss = loss
            bestW = W

#a naive way to evaluate gradient by using limit definition of derivative
def eval_numerical_gradient(f, x):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """

    fx = f(x) # evaluate function value at original point
    grad = np.zeros(x.shape)
    h = 0.00001

    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # evaluate function at x+h
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h # increment by h
        fxh = f(x) # evalute f(x + h)
        x[ix] = old_value # restore to previous value (very important!)

        # compute the partial derivative
        grad[ix] = (fxh - fx) / h # the slope
        it.iternext() # step to next dimension

    return grad

#different activation functions
def activation_functions(x,w1,b1,w2,b2,alpha):
    def sigmoid(x):
        return 1 / (1 + np.exp(-X))
    
    def tanh(x):
        return math.tanh(x)
    
    def ReLU(x):
        return max(0,x)
    
    def leaky_ReLU(x):
        return max(0.1*x,x)
    
    def Maxout(x,w1,b1,w2,b2):
        return max(np.dot(w1.T,x) + b1,np.dot(w2.T,x)+b2)
    
    def ELU(x,alpha):
        if x>=0:
            return x
        else:
            return alpha * (np.exp(x) - 1)



#normalization and regularization codes
def batch_normalization(B):
    #B is a minibatch array of x1..xm
    #batch normalization forces the distribution to be gaussian unit variance
    m = len(B)
    mu_B = sum(B)/m

    variance_B = sum([(xi - mu_B)**2 for xi in B])

    epsilon = 1e-3
    normalized_B = [(xi-mu_B)/np.sqrt(variance_B**2 + epsilon) for xi in B]

    gamma = 1e-2
    beta = 1e-4

    #gamma, beta, and epsilon are parameters to be learned
    y_B = gamma*normalized_B + beta