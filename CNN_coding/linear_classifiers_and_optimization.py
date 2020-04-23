import numpy as np

#the loss of each class given input x, output y, prediction weights W
#for the multi class SVM loss function
def SVM_L_i_vectorized(x,y,W):
	scores = W.dot(x) #predictions for each image
	margins = np.maximum(0,scores - scores[y]+1) #math for hinge loss
	margins[y] = 0 #ignore the correct class labels
	loss_i = np.sum(margins) #add all the losses for the incorrect classes
	return loss_i


#used more widely, this performs softmax loss on input x, output y,
#prediction weights W, and a class label index i to return the loss for a certain class
def Softmax_L_vectorized(x,y,W,i):
    scores = W.dot(x) #classic example of a score function
    scores = np.exp(scores) #exponentiate each term
    scores = scores/np.sum(scores) #normalize each term
    L = -np.log(scores) #take the log
    return L[i]