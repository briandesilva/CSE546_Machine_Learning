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


