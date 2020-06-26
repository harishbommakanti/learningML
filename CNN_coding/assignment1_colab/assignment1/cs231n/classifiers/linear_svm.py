from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
              continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
              #there is loss
              #look at notion notes, derivation is there
              #since loss is W^Tx, dLoss = x. so if theres a margin, all pts of the gradient vector -= the curr X[i]
              dW[:,y[i]] -= X[i]
              dW[:,j] -= X[i]
              loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * W #also look back through notion notes

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]

    scores = X.dot(W)
    #np.arange is basically indices, y is the correct label index --> get the score at each index for 0..numTrain
    correct_class_score = scores[np.arange(num_train), y]

    #correct_class_score is now of shape N
    #correct_class_score[:,np.newaxis] makes it into a Nx1 vector
    margins = np.maximum(0, scores - correct_class_score[:, np.newaxis] + 1)
    margins[np.arange(num_train), y] = 0 #ignore loss of correct labels
    loss = np.sum(margins)
    
    #regularization
    loss /= num_train
    loss += 0.5 * reg * np.sum(W.T.dot(W))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #since loss is W^Tx, dLoss = x. so if theres a margin, all pts of the gradient vector -= the curr X[i]
    X_mask = np.zeros(margins.shape) #a temp array basically
    X_mask[margins > 0] = 1 #basically a piecewise function now, only 0 or 1 in X_mask

    count = np.sum(X_mask, axis=1) #count counts up the # of 1s, or datapoints with lots of loss
    X_mask[np.arange(num_train), y] = -count #subtract the number of counts
    dW = X.T.dot(X_mask)
    dW /= num_train
    dW += np.multiply(W, reg)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
