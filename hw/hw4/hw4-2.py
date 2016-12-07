# CSE 546 Homework 4
# Problem 2: Neural Nets and Backprop

# Brian de Silva
# 1422824

import numpy as np
from mnist import MNIST
import os
import time
import matplotlib.pyplot as plt
import matplotlib


t1 = time.time()

# Import the data
mndata = MNIST('../hw1/mnist')
train_img, train_label = mndata.load_training()
test_img, test_label = mndata.load_testing()

# Convert to numpy arrays
train_img = np.array(train_img)
train_label = np.array(train_label,dtype=int)
test_img = np.array(test_img)
test_label = np.array(test_label,dtype=int)

numClasses = 10

# ----------------------------------------------------------------------------------------
# 							Function defintions
# ----------------------------------------------------------------------------------------


# Neural network class with tanh activation functions
class NN:
	# Constructor where weights are pre-specified
	def __init__(self,W_i,W_h):
		self.W_i = W_i			# Input weights
		self.W_h = W_h			# Hidden layer weights

	# Constructor where size of network is specified in a tuple
	# n = (input dim, # hidden nodes, numClasses)
	# and scaling factor is set in a tuple Scalefact:
	# Scalefact = (scale factor for W_i, factor for W_h)
	def __init__(self,n,scaleFact=None):
		if scaleFact==None:
			scaleFact = (np.sqrt(n[0]),n[0])
		self.W_i = np.random.randn(n[1],n[0]) / scaleFact[0]
		self.W_h = np.random.randn(n[2],n[1]) / scaleFact[1]


	# Generates output from one forward pass through the neural net
	# Data should be contained in COLUMNS of X
	def forwardPass(self,X):
		self.hz = self.W_i.dot(X)				# Hidden layer input
		self.ha = np.tanh(self.hz)				# Hidden layer activation
		self.oa = self.W_h.dot(self.ha)			# Output layer activation

		# # Compute softmax probabilities
		# k = self.W_h.shape[0]+1					# Number of classes
		# probs = np.empty((k,X.shape[1]))
		
		# # Precompute some quantities
		# ex = np.exp(self.ys)
		# denom = -np.log(1 + np.sum(ex,0))

		# # Compute all the probabilities
		# probs[0,:] = np.exp(denom)
		# for j in range(1,k):
		# 	probs[j,:] = np.exp(self.ys[j-1,:] + denom)
		# return probs

	# Computes the gradient of the square loss wrt the weights in the NN
	# Y should be a matrix with each column a length 10 one-hot vector of class labels
	# Y should have already been sampled
	def backProp(self,X,Y,sampleIndices):
		dW_h = -(Y - self.oa[:,sampleIndices])       				# note: activation function is identity
		dW_i = self.W_h.T.dot(dW_h) * (1 - (self.hz[:,sampleIndices]) ** 2)		# Sech^2(x) = 1 - Tanh^2(x)
		return (dW_h.dot(self.ha[:,sampleIndices].T),dW_i.dot(X.T))


	# Computes the gradient of the square loss wrt the weights in the NN
	# Y should be a matrix with each column a length 10 one-hot vector of class labels
	# Y should have already been sampled
	# steps := (stepsize for W_i, stepsize for W_h)
	def backPropFull(self,X,Y,steps):
		dW_h = -(Y - self.oa)       				# note: activation function is identity
		dW_i = self.W_h.T.dot(dW_h) * (1 - (self.hz) ** 2)		# Sech^2(x) = 1 - Tanh^2(x)
		self.W_h -= steps[1] * dW_h
		self.W_i -= steps[0] * dW_i
		# return (dW_h.dot(self.ha.T),dW_i.dot(X.T))


	# Store the weights in files in case we need to re-use them
	def checkpoint(self,fnamePrefix,fnameSuffix=""):
		np.save(fnamePrefix + "W_i" + fnameSuffix,self.W_i)
		np.save(fnamePrefix + "W_h" + fnameSuffix,self.W_h)


	# Compute the mean square loss
	def squareLoss(self,X,Y):
		self.forwardPass(X)
		






# -----------------------------------------------------------------------------------------

# Compute SVD
if os.path.isfile("V.npy"):
	V = np.load("V.npy")
else:
	(U,S,V) = np.linalg.svd(train_img,False)
	np.save("V",V)

# **Note** This V is actually V.H in the traditional SVD: X = U*S*V.H

# Project onto the first 50 singular vectors
trainProj = train_img.dot(V[:50,:].T)
testProj = test_img.dot(V[:50,:].T)
# Note: to get true projection, do trainProj.dot(V[:50,:])

# Create binary labels
trainBinLabels = np.zeros((10,len(train_label)))
labVals = range(10)
for k in range(len(train_label)):
	trainBinLabels[:,k] = train_label[k] == labVals


# Set parameters
# --------------------------------------------------------------------------
nClasses = 10		# Number of classes
reg = 0.			# Regularization parameter
batchSize = 10		# Batch size for SGD
numEpochs = 10		# Number of epochs to go through before terminating
# --------------------------------------------------------------------------

print "Time elapsed during setup: %f" %(time.time() - t1)



# Initialize the neural net
nn = NN((50,500,10))
nn.forwardPass(trainProj.T)

# TODO: figure out scaling for initial weights


# Gradient descent step
step = 1.e-6

# For loop:

# Get new permutation of the inputs
N = train_img.shape[0]
sampleIndices = np.random.permutation(N)

for subIt in range(0, N / batchSize):
	
	# Forward pass
	nn.forwardPass(trainProj[sampleIndices,:].T)

	# Perform backpropagation and update weights with an SGD step
	nn.backPropFull(trainProj[sampleIndices,:].T,trainBinLabels[:,sampleIndices])
	# (dW_h,dW_i) = nn.backPropFull(trainProj[sampleIndices,:].T,trainBinLabels[:,sampleIndices])

	# # Take gradient descent step (build this into backprop?)
	# nn.W_h -= step * dW_h
	# nn.W_i -= step * dW_h