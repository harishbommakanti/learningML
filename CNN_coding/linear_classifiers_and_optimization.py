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

#leave blank for now, just proof of concept
def compute_gradient(x):
    return 0

#Fancier methods of gradient descent below, they don't get stuck at local optima etc.
def SGD_and_Momentum(x):
    #the idea is to have a running 'velocity' to jump over local optima
    vx = 0
    rho = 1e-2 #hyperparameter
    learning_rate = 1e-3 #hyperparameter

    while True:
        dx = compute_gradient(x)
        vx = rho * vx + dx
        x += learning_rate * vx


#advantage over regular momentum update is that it
#gives an idea of the local landscape, aka gradient at new x instead of old x
def Nesterov(x):
    rho = 1e-2 #hyperparameter
    learning_rate = 1e-3 #hyperparameter
    v = 0

    while True:
        dx = compute_gradient(x)
        old_v = v
        v = rho * v - learning_rate * dx #look forward
        x += -rho * old_v + (1 + rho) * v #update parameter vector, weighted diff between next velocity and curr velocity


#another optimization method, not too common: keeps running sum of grad_squared
def Adagrad(x):
    grad_squared = 0
    learning_rate = 1e-3 #hyperparameter

    while True:
        dx = compute_gradient(x)
        grad_squared += dx*dx #keep running sum of square gradients

        x -= learning_rate * dx / np.sqrt(grad_squared) + 1E-7 #helps to accelerate movement along slow dimensions, slow down movement along 'height of taco shells'
    
    #however as time goes on grad_square increases over time --> bad for NNs


#variation of AdaGrad where it adresses problem of slowing down where you might not want too
def RMSProp(x):
    grad_squared = 0
    decay_rate = 1e-2 #hyperparameter
    learning_rate = 1e-3 #hyperparameter

    while True:
        dx = compute_gradient(x)
        grad_squared = decay_rate * grad_squared + (1 - decay_rate) * dx * dx
        x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
            #1e-7 cuz you never want to divide by 0


#Instructor's favorite, 'default choice' for gradient descent
def Adam(x):
    #a mix of Momentum, bias correction, and RMSProp

    first_moment = 0
    second_moment = 0
    num_iterations = 1000
    learning_rate = 1e-4 #hyperparameter

    #hyperparameters which have their ideal implementations below
    beta1 = 0.9
    beta2 = 0.99
    alpha = 1e-3 #or 5e-4

    for t in range(num_iterations):
        dx = compute_gradient(x)

        #MOMENTUM STEP BELOW
        first_moment = beta1 * first_moment + (1-beta1) * dx

        #RMSPROP step 1
        second_moment = beta2 * second_moment + (1-beta2) * dx * dx

        #Bias correction below
        first_unbias = first_moment / (1 - beta1 ** t)
        second_unbias = second_moment / (1 - beta2 ** t)

        #Last step of RMSProp
        x -= learning_rate * first_unbias / (np.sqrt(second_unbias) + 1e-7)