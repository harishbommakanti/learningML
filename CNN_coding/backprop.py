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