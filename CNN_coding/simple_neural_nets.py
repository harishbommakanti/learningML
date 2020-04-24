import numpy as np
class Neuron: #very very loose model of how sigmoid is an activation function
	def neuron_tick(self,inputs):
		cell_body_sum = np.sum(inputs * self.weights + self.bias)
		firing_rate = 1/(1 + np.exp(-cell_body_sum))
		return firing_rate