import numpy as np

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0] #records the number of rows
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype) #with num_test rows of zeroes

    # loop over all test rows
    for i in range(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      
      #distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]))) #another way of getting the distances, sqrt of sum of squares (standard)
      
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred #will have a prediction index Ypred[i] for each row/test image at i

def load_CIFAR10(fname):
    xtr,ytr,xte,yte = [],[],[],[]
    return xtr,ytr,xte,yte

def runVanillaNearestNeighbors():
    Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10/') # a magic function we provide
    # flatten out all images to be one-dimensional
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes an array of size 50000 x 3072 elements
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes an array of size 10000 x 3072 elements

    nn = NearestNeighbor() # create a Nearest Neighbor classifier class
    nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
    Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
    # and now print the classification accuracy, which is the average number
    # of examples that are correctly predicted (i.e. label matches)
    print('accuracy: %f' % ( np.mean(Yte_predict == Yte)))

#runVanillaNearestNeighbors()