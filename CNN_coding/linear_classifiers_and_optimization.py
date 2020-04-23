import numpy as np

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