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


# Neural network class using tanh activation function at hidden layer
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

	# # Computes the gradient of the square loss wrt the weights in the NN
	# # Y should be a matrix with each column a length 10 one-hot vector of class labels
	# # Y should have already been sampled
	# def backPropPartial(self,X,Y,sampleIndices):
	# 	dW_h = -(Y - self.oa[:,sampleIndices])       				# note: activation function is identity
	# 	dW_i = self.W_h.T.dot(dW_h) * (1 - (self.hz[:,sampleIndices]) ** 2)		# Sech^2(x) = 1 - Tanh^2(x)
	# 	return (dW_h.dot(self.ha[:,sampleIndices].T),dW_i.dot(X.T))


	# Computes the gradient of the square loss wrt the weights in the NN
	# Y should be a matrix with each column a length 10 one-hot vector of class labels
	# steps := (stepsize for W_i, stepsize for W_h)
	def backProp(self,X,Y,steps):
		N = X.shape[1]
		dW_h = -(Y - self.oa) / N       				# note: activation function is identity
		dW_i = self.W_h.T.dot(dW_h) * (1 - (self.hz) ** 2) / N		# Sech^2(x) = 1 - Tanh^2(x)
		self.W_h -= steps[1] * dW_h.dot(self.ha.T)
		self.W_i -= steps[0] * dW_i.dot(X.T)
		# return (dW_h.dot(self.ha.T),dW_i.dot(X.T))

	# A function that returns the gradients of the two weight matrices
	def getGrads(self,X,Y):
		self.forwardPass(X)
		N = X.shape[1]
		dW_h = -(Y - self.oa) / N       				# note: activation function is identity
		dW_i = self.W_h.T.dot(dW_h) * (1 - (self.hz) ** 2) / N		# Sech^2(x) = 1 - Tanh^2(x)
		return (dW_i.dot(X.T), dW_h.dot(self.ha.T))


	# Store the weights in files in case we need to re-use them
	def checkpoint(self,fnamePrefix,fnameSuffix=""):
		np.save(fnamePrefix + "W_i" + fnameSuffix,self.W_i)
		np.save(fnamePrefix + "W_h" + fnameSuffix,self.W_h)


	# Compute the mean square loss
	# Y should be a matrix with each column a length 10 one-hot vector of class labels
	def squareLoss(self,X,Y):
		self.forwardPass(X)
		return np.mean(np.sum((Y - self.oa)**2,0))

	# Compute the 0/1 loss
	# Y should be a vector of class labels
	def z1Loss(self,X,Y):
		self.forwardPass(X)
		return (1.0 * np.count_nonzero(np.argmax(self.oa,0) - Y)) / Y.shape[0]

	# Compute mean square loss and 0/1 loss
	# Y should be a matrix with each column a length 10 one-hot vector of class labels
	def getLoss(self,X,Y):
		self.forwardPass(X)

		# Compute square loss
		sql = np.mean(np.sum((Y - self.oa)**2,0))

		# Compute 0/1 loss
		z1l = 1. * np.count_nonzero(np.argmax(Y,0) != np.argmax(self.oa,0)) / Y.shape[1]

		return (sql,z1l)



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
testBinLabels = np.zeros((10,len(test_label)))
labVals = range(10)
for k in range(len(train_label)):
	trainBinLabels[:,k] = (train_label[k] == labVals)
	testBinLabels[:,k] = (test_label[k] == labVals)


# Set parameters
# --------------------------------------------------------------------------
nClasses = 10			# Number of classes
reg = 0.				# Regularization parameter
batchSize = 10			# Batch size for SGD
numEpochs = 10			# Number of epochs to go through before terminating
nnSize = (50,500,10)	# Number of nodes in each layer of NN (input, hidden, output)
std = (np.sqrt(np.mean(trainProj**2)),np.sqrt(50))	# Initial weight standard deviations
ckptStr = "init_test2.1"	# String used in checkpointing filename

print "Time elapsed during setup: %f" %(time.time() - t1)
# --------------------------------------------------------------------------


# 
# Train the network
# 

# Initialize the neural net
nn = NN(nnSize,std)
nn.forwardPass(trainProj.T)

# Rescale the weights so that E(Yhat) ~ E(Y) / 10
# Note: E(Y) = 1/10
nn.W_h /= (100*np.mean(nn.oa))
nn.W_i /= (100*np.mean(nn.oa))

# Create vectors in which to store the loss
trainLoss = np.zeros((2,2 * numEpochs + 1))		# First row is sq loss, 2nd is 0/1
testLoss = np.zeros((2,2 * numEpochs + 1))		# First row is sq loss, 2nd is 0/1


# SGD
step = 1.e-4				# Learning rate
N = train_img.shape[0]

# Loop over epochs
for it in range(numEpochs):
	t2 = time.time()

	# Get new sampling order for this epoch
	sampleIndices = np.random.permutation(N)

	# Compute losses
	trainLoss[:,2*it] = nn.getLoss(trainProj.T,trainBinLabels)
	testLoss[:,2*it] = nn.getLoss(testProj.T,testBinLabels)

	# Print losses
	print "Square loss after %d epochs: %f"%(it,trainLoss[0,2*it])
	print "0/1 loss after %d epochs: %f"%(it,trainLoss[1,2*it])

	# Take multiple gradient steps for half an epoch
	for subIt in range((N / batchSize) / 2):
		# Get indices for this iteration
		currIndices = sampleIndices[subIt*batchSize:(subIt+1)*batchSize]

		# Forward pass
		nn.forwardPass(trainProj[currIndices,:].T)

		# Update weights
		nn.backProp(trainProj[currIndices,:].T,trainBinLabels[:,currIndices],(step,step))



	# Compute and print losses after half an epoch
	trainLoss[:,2*it+1] = nn.getLoss(trainProj.T,trainBinLabels)
	testLoss[:,2*it+1] = nn.getLoss(testProj.T,testBinLabels)
	print "Square loss after %d 1/2 epochs: %f"%(it,trainLoss[0,2*it+1])
	print "0/1 loss after %d 1/2 epochs: %f"%(it,trainLoss[1,2*it+1])

	# Check the norms of the full gradients
	(gi,gh) = nn.getGrads(trainProj.T,trainBinLabels)
	print "Orders of gradients W_i, W_h: %f, %f"%(np.sqrt(np.sum(gi**2)),np.sqrt(np.sum(gh**2)))

	# Finish the epoch
	for subIt in range((N / batchSize) / 2, N / batchSize):
		# Get indices for this iteration
		currIndices = sampleIndices[subIt*batchSize:(subIt+1)*batchSize]

		# Forward pass
		nn.forwardPass(trainProj[currIndices,:].T)

		# Update weights
		nn.backProp(trainProj[currIndices,:].T,trainBinLabels[:,currIndices],(step,step))

	


# # For loop:

# # Get new permutation of the inputs
# N = train_img.shape[0]
# sampleIndices = np.random.permutation(N)

# for subIt in range(0, N / batchSize):
	
# 	# Forward pass
# 	nn.forwardPass(trainProj[sampleIndices,:].T)

# 	# Perform backpropagation and update weights with an SGD step
# 	nn.backPropFull(trainProj[sampleIndices,:].T,trainBinLabels[:,sampleIndices])
# 	# (dW_h,dW_i) = nn.backPropFull(trainProj[sampleIndices,:].T,trainBinLabels[:,sampleIndices])

# 	# # Take gradient descent step (build this into backprop?)
# 	# nn.W_h -= step * dW_h
# 	# nn.W_i -= step * dW_h