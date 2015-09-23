"""
	#-------------------------------------------------------------------------------
neuralnet_main.py
This is the main module of neural neural network of the stochastic gradient descent algorithm
"""
import random
import numpy as np

class NeuralNet():
	def __init__(self,ls):
		"""Random initialization for the weights and biases of neural networking with respect
		to the structure of neural network characterized by ls(layer sizes)
		here we use the sigmoid function sigma(z) = 1/(1+exp(-z)), z = wx+b"""
		self.Nls = len(ls)
		self.ls = ls
		self.bs = [np.random.randn(i,1) for i in ls[1:]]
		self.ws = [np.random.randn(i,j) for i, j in zip(ls[1:], ls[:-1])]		
	#-------------------------------------------------------------------------------
	def feedforward(self,a):
		""" This is the feedforward function computing the out put of the network regarding the sigmoid function """
		for w,b in zip(self.ws,self.bs):
			z = np.dot(w,a)+b
			a = sigmoid_vec(z)
		return a

	#-------------------------------------------------------------------------------
	def Stochastic_GD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
		""" Train the neural network using mini-batch stochastic gradient descent. The "training data" is a list of 
		tuples "(x,y)" representing the training inputs and the desired outputs. The other non-optional parameters
		are self-explanatory. If "test_data" is provided then the network will be evaluated against the test data
		after each epoch, and partial progress printed out. This is useful fro tracking progress, but slows things
		down substantially."""
		if test_data:
			n_test = len(test_data)
		n_train = len(training_data)
		for k in xrange(epochs):
			random.shuffle(training_data)
			mini_batches = [training_data[j:j+mini_batch_size]
				for j in xrange(0,n_train, mini_batch_size)]

	#-------------------------------------------------------------------------------
	def back_propagate(self,x,y):
		"""Return a tuple "(nabla_bs, nabla_ws)" representing the gradient for the cost function C_x. "nabla_bs" and 
		"nabla_ws" are layer-by-layer lists of numpy arrays, similar "self.bs" and "self.ws"
		"""
		nabla_bs = [np.zeros(b.shape) for b in self.bs]
		nabla_ws = [np.zeros(w.shape) for w in self.ws]
		# Feedforward
		actv = x
		actvs = [x]
		zs = []
		for b, w in zep(self.bs,self.ws)
			z = np.dot(w,actv)+b
			zs.append(z)
			actv = sigmoid_vec(z)
			actvs.append(actv)
		# Backward pass
		delta = self.cost_der(actvs[-1],y)*sigmoid_der_vec(zs[-1])
		nabla_bs[-1] = delta
		nabla_ws[-1] = np.dot(delta,activs[-2].transpose())


	#-------------------------------------------------------------------------------
	def cost_der(self,out_actv,y):
		"""Return the vector of partial dertivatives \partial C_x/\partial a for the output activations."""
		return (out_actv-y)


		
			
			

#---------------------------------------------------------------------------------------
#Miscellaneous functions
def sigmoid(z):	
	""" This is the sigmoid function """
	sigma = 1.0/(1.0+np.exp(-z))
	return sigma
""" Vectorize the output of the sigmoid function when input is a vector """
sigmoid_vec = np.vectorize(sigmoid)
#
def sigmoid_der(z):	
	""" Take the derivative of the sigmoid function """
	sigma_prime = sigmoid(z)*(1.0-sigmoid(z))
	return sigma
""" Vectorize the output of the derivative of the sigmoid function when input is a vector"""
sigmoid_der_vec = np.vectorize(sigmoid_der)



	 
	  
	
	
