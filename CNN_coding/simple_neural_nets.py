import numpy as np
class Neuron: #very very loose model of how sigmoid is an activation function
	def neuron_tick(self,inputs):
		cell_body_sum = np.sum(inputs * self.weights + self.bias)
		firing_rate = 1/(1 + np.exp(-cell_body_sum))
		return firing_rate


#writing a very simple 20 line version of a 2 layer neural net
# architecture: x -> W1 -> h -> W2 -> output
def simple_two_layer_net():
    from numpy.random import randn

    N, D_in, H, D_out = 62,1000,100,10 #dimensions
    x,y = randn(N, D_in), randn(N, D_out)
    w1,w2 = randn(D_in,H), randn(H,D_out) #notice the dimensions and connection to matrix multiplication

    for t in range(2000): #for 2000 iterations
        h = 1 / (1 + np.exp(-x.dot(w1)))
        y_pred = h.dot(w2) #the rightmost layer
        loss = np.square(y_pred - y).sum() #normal sum of square loss
        
        #compute the gradient from the back, which is y_pred
        #look at the h function above to understand the specific calculations, is sigmoid(x.dot(W1)) which makes sense
        grad_y_pred = 2 * (y_pred-y) #assumes L2, so gradient is 2 * y_diff
        grad_w2 = h.T.dot(grad_y_pred)
        grad_h = grad_y_pred.dot(w2.T)
        grad_w1 = x.T.dot(grad_h * h * (1-h))

        #adjust the weights
        w1 -= 1e-4 * grad_w1
        w2 -= 1e-4 * grad_w2


#following method is just abstract code for the view of neurons forming a layer,
#input layer --> hidden layer 1 --> hidden layer 2 --> output layer
def neurons_as_a_layer_view():
    f = lambda x: 1/(1 + np.exp(-x)) #activation function, just ur regular sigmoid
    
    #x has 3 neurons (3x1), hid layer 1 has 4 (4x1), hid layer 2 has 4 (4x1), output is 1 (1x1)
    x = np.random.randn(3,1) #random input vector of size 3x1
    W1 = np.random.randn(4,3) #dimensions to make h of size 4x1
    b1 = np.random.randn(4,1) #bias of size 4x1
    
    W2 = np.random.randn() #dimensions to make h2 4x1 would just be a 1x1 matrix (1 #)
    b2 = np.random.randn(4,1) #bias of size 4x1
    W3,b3 = np.random.randn(),np.random.randn() #final result weighting and bias would be 1# too

    h1 = f(np.dot(W1,x)+b1) #first hidden layer 4x1
    h2 = f(np.dot(W2,h1)+b2) #second hidden layer 4x1
    out = np.dot(W3,h2)+b3 #output layer

simple_two_layer_net()
neurons_as_a_layer_view()