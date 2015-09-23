import numpy as np
def sigmoid(z):
	sigma = 1.0/(1.0+np.exp(-z))
	return sigma	
