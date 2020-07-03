import numpy as np
import math

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


#normal zero-centering and normalization
def preprocess(X):
    #zero center
    X -= np.mean(X,axis=0)

    #normalizing data
    X /= np.std(X,axis=0)

#PCA, another form of preprocessing by computing covariance matrix
def PCA(X):
    #zero center
    X -= np.mean(X,axis=0)

    #get covariance matrix
    cov = np.dot(X.T,X) / X.shape[0]

    #do singular value decomposition
    U,S,V = np.linalg.svd(cov)

    #can decorrelate data
    Xrot = np.dot(X,U)


    #final step of discarding dimensions with negligible variance
    Xrot_reduced = np.dot(X, U[:,:100])

    return Xrot_reduced