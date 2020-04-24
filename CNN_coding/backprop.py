import numpy as np
def simple_chain_rule_ex():
    #f(x,y,z) = (x+y)z, say q = x+y

    # set some inputs
    x = -2; y = 5; z = -4

    # perform the forward pass
    q = x + y # q becomes 3
    f = q * z # f becomes -12

    # perform the backward pass (backpropagation) in reverse order:
    # first backprop through f = q * z
    dfdz = q # df/dz = q, so gradient on z becomes 3
    dfdq = z # df/dq = z, so gradient on q becomes -4
    # now backprop through q = x + y
    dfdx = 1.0 * dfdq # dq/dx = 1. And the multiplication here is the chain rule!
    dfdy = 1.0 * dfdq # dq/dy = 1

def sigmoid_chain_rule_ex():
    #f(w,x) = 1 / (1 + np.exp(-(w0x0 + w1x1 + w2)))

    w = [2,-3,-3] # assume some random weights and data
    x = [-1, -2]

    # forward pass
    dot = w[0]*x[0] + w[1]*x[1] + w[2]
    f = 1.0 / (1 + np.exp(-dot)) # sigmoid function

    # backward pass through the neuron (backpropagation)
    ddot = (1 - f) * f # gradient on dot variable, using the sigmoid gradient derivation
    dx = [w[0] * ddot, w[1] * ddot] # backprop into x
    dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot] # backprop into w
    # we're done! we have the gradients on the inputs to the circuit

def complex_backprop_example():
    #sigma(x) = 1/1+e^(-x)
    #f(x,y) = (x + sigma(y)) / ((sigma(x) + (x+y)^2)
    
    x = 3 # example values
    y = -4

    # forward pass
    sigy = 1.0 / (1 + np.exp(-y)) # sigmoid in numerator   #(1)
    num = x + sigy # numerator                               #(2)
    sigx = 1.0 / (1 + np.exp(-x)) # sigmoid in denominator #(3)
    xpy = x + y                                              #(4)
    xpysqr = xpy**2                                          #(5)
    den = sigx + xpysqr # denominator                        #(6)
    invden = 1.0 / den                                       #(7)
    f = num * invden # done!                                 #(8)


    #backward pass
    #backprop of f = num * invden: gradient switcher
    dnum = invden
    dinvden = num

    #backprop of invden = 1.0/den: local gradient of 1/den = -1/(den^2), multiply that by df/d(invden)
    dden = (-1.0 / (den**2)) * dinvden

    #backprop of den = sigx + xpysqr: pass in 1 cuz +, end up with 1 * df/d(den)
    dsigx = 1 * dden
    dxpysqr = 1 * dden

    #backprop of dxpysqr = xpy**2, local gradient is 2*xpy, prev gradient is df/d(xpysqr)
    dxpy = (2 * dxpy) * dxpysqr

    #backprop of xpy = dx + dy: passess 1 as a gradient cuz +, 1 + df/d(xpy)
    dx = 1*dxpy
    dy = 1*dxpy

    #backprop of sigx = 1/(1+math.exp(-x)): quotient rule
    dx += ((1 - sigx)*sigx) * dsigx # += as gradients add at branching

    #backprop of num = x + sigy: df/dx += 1 * dnum (+ gate), df/dsigy += 1 * dsigy
    dx += 1*dnum
    dsigy = 1*dnum

    #backprop of sigy = 1/(1+matht.exp(-y))
    dy += ((1-sigy)*sigy)*dsigy

    #go back and put df in front of every d_ for this to make sense, simple df/dx = df/d_ * d_/dx

def matrix_multiplication_backprop_ex():
    #gradients for multiplying matrices

    #forward pass (simple loss computation)
    W = np.random.randn(5,10) # a 5x10 matrix
    X = np.random.randn(10,3) # a 10x3 matrix
    D = W.dot(X) # a 5x3 matrix

    # backprop
    dD = np.random.randn(*D.shape)
    dW = dD.dot(X.T) #5x3 * 3x10 --> 5x10, shape of W. dW has same shape as W, good
    dX = W.T.dot(dD) #10x5 * 5x3 --> 10x3, shape of X. dX has same shape as X, good
    #its the same type of thing for multiplication regularly, gradient switching

    #USE DIMENSION ANALYSIS: check if d_ has same shape as _, the Jacobian should


#modularized gate and computational graph from the slides listed below
#just really rough implementations and their analog to real world deep learning frameworks, not really syntactically accurate

class ComputationalGraph(object):
	def forward(self,inputs,loss):
		#1: [pass inputs to input gates]
		#2: forward through the comp. graph
		for gate in self.graph.nodes_topologically_sorted():
			gate.forward()
		return loss #final gate in graph outputs the loss

	def backward(self,inputs_gradients):
		for gate in reversed(self.graph.nodes_topologically_sorted()):
			gate.backward() #little piece of back prop (chain rule applied)
		return inputs_gradients

	#keep in mind gate.backward() =/= ComputationalGraph.backward(),
	# in this model, gates are like inner objects of the whole graph.
	# no recursion going on here

class MultiplyGate(object):
    def forward(self,x,y):
        z = x*y
        self.x = x #caching the values to not lose them
        self.y = y
        return z #makes sense, just return the value

    #following is the partial L/partial z, given the most recent gradient dz from the right (in backprop)	
    def backward(self,dz):
        dx = self.y * dz #[dz/dx * dL/dz] scaling it by values of other branch like gradient switching, remember
        dy = self.x * dz #[dz/dy * dL/dz]
        return [dx,dy] #has [partial L/partial x, partial L/partial y]