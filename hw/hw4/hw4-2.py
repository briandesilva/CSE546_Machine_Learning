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

	# Constructor where only size of network is specified in a tuple
	# n = (input dim, # hidden nodes, numClasses)
	def __init__(self,n):
		self.W_i = np.random.randn(n[1],n[0]) / np.sqrt(n[0])
		self.W_h = np.random.randn(n[2],n[1]) / n[0]


	# Generates output from one forward pass through the neural net
	# Data should be contained in COLUMNS of x
	def forwardPass(self,x):
		self.hs = np.tanh(self.W_i.dot(x))		# Input activation
		self.ys = self.W_h.dot(hs)				# Output activation

		# # Compute softmax probabilities
		# k = self.W_h.shape[0]+1					# Number of classes
		# probs = np.empty((k,x.shape[1]))
		
		# # Precompute some quantities
		# ex = np.exp(self.ys)
		# denom = -np.log(1 + np.sum(ex,0))

		# # Compute all the probabilities
		# probs[0,:] = np.exp(denom)
		# for j in range(1,k):
		# 	probs[j,:] = np.exp(self.ys[j-1,:] + denom)
		# return probs











# -----------------------------------------------------------------------------------------

# Compute SVD
if os.path.isfile("U.npy"):
	V = np.load("V.npy")
else:
	(U,S,V) = np.linalg.svd(train_img,False)
	np.save("V",V)

# **Note** This V is actually V.H in the traditional SVD: X = U*S*V.H

# Project onto the first 50 singular vectors
trainProj = train_img.dot(V[:50,:].T)
testProj = test_img.dot(V[:50,:].T)
# Note: to get true projection, do trainProj.dot(V[:50,:])


# Set parameters
# --------------------------------------------------------------------------
nClasses = 10		# Number of classes
reg = 0.			# Regularization parameter
batchSize = 10		# Batch size for SGD
numEpochs = 10		# Number of epochs to go through before terminating
# --------------------------------------------------------------------------

print "Time elapsed during setup: %f" %(time.time() - t1)


