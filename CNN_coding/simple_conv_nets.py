import numpy as np
def creating_activation_maps():
    X = np.random.randn(11,11,4) #random input of size 11x11x4
    F = 5 #filter of size 5x5x{number of filters}
    D = 2 #number of filters/depth of output volume is 2
    S = 2 #skip every other pixel
    P = 0 #no padding

    V = np.empty((F,F,D))
    W0 = np.random.randn(F,F,1) #filter is just a 2d square
    W1 = np.random.randn(F,F,1)
    b0 = np.random.randn() #bias is just 1 value to add to the pixel val of the output volume
    b1 = np.random.randn()

    #activation map examples for the first conv layer
    #W0 is the first filter
    V[0,0,0] = np.sum(X[:F,:F,:] * W0) + b0
    V[0,0,0] = np.sum(X[:5,:5,:] * W0) + b0 
    V[1,0,0] = np.sum(X[2:7,:5,:] * W0) + b0 
    V[2,0,0] = np.sum(X[4:9,:5,:] * W0) + b0 
    V[3,0,0] = np.sum(X[6:11,:5,:] * W0) + b0

    #some activation map examples for the second conv layer, for 2nd depth
    V[0,0,1] = np.sum(X[:5,:5,:] * W1) + b1 
    V[1,0,1] = np.sum(X[2:7,:5,:] * W1) + b1 
    V[2,0,1] = np.sum(X[4:9,:5,:] * W1) + b1 
    V[3,0,1] = np.sum(X[6:11,:5,:] * W1) + b1 
    V[0,1,1] = np.sum(X[:5,2:7,:] * W1) + b1
    V[2,3,1] = np.sum(X[4:9,6:11,:] * W1) + b1

    #just showed a few examples to see how it works
    print(V)

creating_activation_maps() #works at least for random values