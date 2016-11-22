# CSE 546 Homework 3
# Problem 2: MNIST revisited

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


# #-------------------------------------------------------------------------------------------------------------
# #											Part 2.1 - Least Squares
# #-------------------------------------------------------------------------------------------------------------


# 
# Define some functions that we will need
# 

# ---RBF Kernel---
# 
# Returns the feature vector consisting of entries which are RBF kernels of x and
# other data points in X. sigma parameterizes what counts as a "large" distance. Note
# that as sigma goes to 0, approaches an elementwise indicator function
def RBFVec(x,X,X2,sigma):
	return np.exp(-(X2 + np.sum(x**2) - 2. * X.dot(x)) / ((2. * sigma ** 2)))


# ---Fourier Kernel---
# 
# Returns the feature vector of Fourier features
def FRBFVec(x,V,sigma):
	return np.sin(V.dot(x) / sigma)

# ---Stochastic Gradient---
# 
# Computes an approximate gradient of the square loss function using sample points specified
# by sampleIndices
def stochGradient(Y,X,V,W,reg,sampleIndices,sigma):
	output = np.zeros(np.prod(W.shape))
	labels = np.arange(0,10)
	for ii in range(0,sampleIndices.shape[0]):
		rbf = FRBFVec(X[sampleIndices[ii],:],V,sigma)
		output += np.kron(W.dot(rbf) - (Y[sampleIndices[ii]]==labels).astype(int),rbf)
	return 2. * output.reshape(W.shape) + reg * W

# ---0/1 loss---
# 
# Computes the 0/1 loss
def z1Loss(Y,X,V,w,sigma):
	# Get predictions for each point
	model_out = np.empty(Y.shape[0])
	for k in range(0,Y.shape[0]):
		model_out[k] = np.argmax(w.dot(FRBFVec(X[k,:],V,sigma)))
	return (1.0 * np.count_nonzero(model_out - Y)) / Y.shape[0]

# ---Square loss---
# 
# square loss for this problem
def squareLoss(Y,X,V,W,reg,sigma):
	total = 0.0
	labels = np.arange(0,10)
	for k in range(0,Y.shape[0]):
		total += np.sum(((labels==Y[k]).astype(int) - W.dot(FRBFVec(X[k,:],V,sigma))) ** 2)
	return total + reg * np.sum(W**2) / 2.

# ---0/1 and square loss---
# 
# Gets the 0/1 and square losses simultaneously so we don't have to recompute feature vectors
# Doe this for both weights and average weights over last epoch
def getLoss(Y,X,V,W,WAvg,reg,sigma):
	model_out = np.empty(Y.shape[0])
	model_outAvg = np.empty(Y.shape[0])
	total = 0.
	totalAvg = 0.
	labels = np.arange(10)
	for k in range(Y.shape[0]):
		rbf = FRBFVec(X[k,:],V,sigma)
		model_out[k] = np.argmax(W.dot(rbf))
		total += np.sum(((labels==Y[k]).astype(int) - W.dot(rbf)) ** 2)
		model_outAvg[k] = np.argmax(WAvg.dot(rbf))
		totalAvg += np.sum(((labels==Y[k]).astype(int) - WAvg.dot(rbf)) ** 2)
	z1_out = (1.0 * np.count_nonzero(model_out - Y)) / Y.shape[0]
	sq_out = total + reg * np.sum(W**2) / 2.
	z1_outAvg = (1.0 * np.count_nonzero(model_outAvg - Y)) / Y.shape[0]
	sq_outAvg = totalAvg + reg * np.sum(WAvg**2) / 2.
	return (sq_out,z1_out,sq_outAvg,z1_outAvg)


# ---Stochastic gradient descent---
# 
# Stochastic gradient descent method (square loss)
# 
#	 -YTrain is an array of labels for training set
#	 -YTest is an array of labels for test set
#	 -XTrain is the data points for training set
#	 -XTest is the data points for test set
#	 -V is the set of vectors used to produce the Fourier features with the FRBF function
#	 -sigma is a parameter in the RBF and FRBF kernels
#	 -reg is the regularization parameter
#	 -nClasses is the number of classes into which the data should be classified
#
def SGD(YTrain,YTest,XTrain,XTest,V,sigma,reg,nClasses,batchSize=100,numEpochs=10,TOL=1.e-5,w=None):
	t3 = time.time()
	if w is None:
		w = np.zeros((nClasses,YTrain.shape[0]))	# Note: this changed from before

	# wOld = np.zeros(w.shape)
	N = XTrain.shape[0]
	num = 1.e-5 / 2.
	step = num
	wAvg = np.zeros(w.shape)

	squareLossTrain = np.zeros(numEpochs + 1)
	squareLossTest = np.zeros(numEpochs + 1)
	z1LossTrain = np.zeros(numEpochs + 1)
	z1LossTest = np.zeros(numEpochs + 1)

	squareLossTrainAvg = np.zeros(numEpochs + 1)
	squareLossTestAvg = np.zeros(numEpochs + 1)
	z1LossTrainAvg = np.zeros(numEpochs + 1)
	z1LossTestAvg = np.zeros(numEpochs + 1)

	# Loop over epochs
	for it in range(0,numEpochs):

		t5 = time.time()
		# Check square loss and 0/1 loss every time we pass through all the points
		(squareLossTrain[it],z1LossTrain[it],squareLossTrainAvg[it],z1LossTrainAvg[it]) = getLoss(YTrain,XTrain,V,w,wAvg,reg,sigma)
		(squareLossTest[it],z1LossTest[it],squareLossTestAvg[it],z1LossTestAvg[it]) = getLoss(YTest,XTest,V,w,wAvg,reg,sigma)

		t6 = time.time()
		print "Time to compute the square loss: %f"%(t6-t5)
		print "Square loss at iteration %d (w): %f"%(it,squareLossTrain[it])
		print "0/1 loss at iteration %d (w): %f"%(it,z1LossTrain[it])
		print "Square loss at iteration %d (wAvg): %f"%(it,squareLossTrainAvg[it])
		print "0/1 loss at iteration %d (wAvg): %f"%(it,z1LossTrainAvg[it])

		# Compute new order in which to visit indices
		sampleIndices = np.random.permutation(N)

		# Zero out wAvg for next epoch
		wAvg = np.zeros(w.shape)

		# Loop over all the points
		for subIt in range(0,N / batchSize):
			# Compute the gradient
			currentIndices = sampleIndices[subIt*batchSize:(subIt+1)*batchSize]
			newGrad = stochGradient(YTrain,XTrain,V,w,reg,currentIndices,sigma)

			# Precompute a possibly expensive quantity
			gradNorm = np.sqrt(np.sum(newGrad**2))

			# Take a step in the negative gradient direction
			step = num / np.sqrt(it+1)
			w = w - step * newGrad
			wAvg += w

			if gradNorm < TOL:
				# Method has converged, so record loss and exit
				(squareLossTrain[it],z1LossTrain[it],squareLossTrainAvg[it],z1LossTrainAvg[it]) = getLoss(YTrain,XTrain,V,w,wAvg,reg,sigma)
				(squareLossTest[it],z1LossTest[it],squareLossTestAvg[it],z1LossTestAvg[it]) = getLoss(YTest,XTest,V,w,wAvg,reg,sigma)

				print "Order of gradient: %f\n" %np.log10(gradNorm)
				wAvg /= (subIt + 1)
				break

		# Compute average weight over previous epoch
		wAvg /= (N / batchSize)
		
		# Print time to complete an epoch
		t9 = time.time()
		print "Time elapsed during epoch %d: %f" %(it,t9-t6)

		# Print out size of the gradient
		print "Order of gradient: %f\n" %np.log10(gradNorm)

	if (it == numEpochs-1):
		print "Warning: Maximum number of iterations reached."

	(squareLossTrain[it+1],z1LossTrain[it+1],squareLossTrainAvg[it+1],z1LossTrainAvg[it+1]) = getLoss(YTrain,XTrain,V,w,wAvg,reg,sigma)
	(squareLossTest[it+1],z1LossTest[it+1],squareLossTestAvg[it+1],z1LossTestAvg[it+1]) = getLoss(YTest,XTest,V,w,wAvg,reg,sigma)

	t4 = time.time()
	print "Time to perform %d epochs of SGD with batch size %d: %f"%(it+1,batchSize,t4-t3)
	
	return (w,wAvg,squareLossTrain[:(it+2)],z1LossTrain[:(it+2)],squareLossTest[:(it+2)],z1LossTest[:(it+2)],squareLossTrainAvg[:(it+2)],z1LossTrainAvg[:(it+2)],squareLossTestAvg[:(it+2)],z1LossTestAvg[:(it+2)])


# -------------------------------------------------------------------------------------

# First we want to project onto the first 50 singular vectors 

# Compute SVD
if os.path.isfile("U.npy"):
	U = np.load("U.npy")
	S = np.load("S.npy")
	V = np.load("V.npy")
else:
	(U,S,V) = np.linalg.svd(train_img,False)
	np.save("U",U)
	np.save("S",S)
	np.save("V",V)

# **Note** This V is actually V.H in the traditional SVD: X = U*S*V.H

# Project onto the first 50 singular vectors
trainProj = train_img.dot(V[:50,:].T)
testProj = test_img.dot(V[:50,:].T)
# Note: to get true projection, do trainProj.dot(V[:50,:])

# Estimate a good value of sigma for RBFVec using the median trick
N = trainProj.shape[0]
numSamples = 100
dists = 0.
inds = np.empty(2)
for k in range(0,numSamples):
	inds = np.random.choice(np.arange(0,N),2,replace=False)	# Get a random pair of data points
	dists += np.sqrt(np.sum((trainProj[inds[0],:] - trainProj[inds[1],:])**2))

# Cheat a little and use the empirical mean distance between points
dists /= numSamples

# Generate or load random vectors for features
if os.path.isfile("feats.npy"):
	feats = np.load("feats.npy")
else:
	feats = np.random.randn(trainProj.shape[0],trainProj.shape[1])
	np.save("feats",feats)
 
# Set parameters
# --------------------------------------------------------------------------
sigma = dists / 2.
nClasses = 10		# Number of classes
reg = 0.			# Regularization parameter
batchSize = 10		# Batch size for SGD
numEpochs = 10		# Number of epochs to go through before terminating
# --------------------------------------------------------------------------

print "Time elapsed during setup: %f" %(time.time() - t1)


# Run the method
(w, wAvg, squareLossTrain, z1LossTrain,
squareLossTest, z1LossTest,
squareLossTrainAvg, z1LossTrainAvg,
squareLossTestAvg, z1LossTestAvg) = SGD(train_label,
										test_label,
										trainProj,
										testProj,
										feats,
										sigma,
										reg,
										nClasses,
										batchSize,
										numEpochs)


# Plot square loss on test and train sets
matplotlib.rcParams.update({'font.size' : 20})
n = len(squareLossTrain)-1
outputCond = train_label.shape[0] / batchSize
its = range(0,n*outputCond+1,outputCond)
plt.figure(1)
plt.plot(its,squareLossTrain,'b-o',its,squareLossTrainAvg,'k--o',
	its,squareLossTest,'r-x',its,squareLossTestAvg,'g--x',linewidth=1.5)
plt.xlabel('Iteration')
plt.ylabel('Square loss')
plt.legend(['Training (w)','Training (wAvg)','Test (w)', 'Test (wAvg)'])

# Plot 0/1 loss on test and train sets
plt.figure(2)
plt.plot(its[1:],z1LossTrain[1:],'b-o',its[1:],z1LossTrainAvg[1:],'k--o',
	its[1:],z1LossTest[1:],'r-x',its[1:],z1LossTestAvg[1:],'g--x',linewidth=1.5)
plt.xlabel('Iteration')
plt.ylabel('0/1 loss')
plt.legend(['Training (w)','Training (wAvg)','Test (w)', 'Test (wAvg)'])
plt.show()

# Print out final 0/1 and square loss
print "Square loss (training): %f" % (squareLossTrainAvg[-1])
print "0/1 loss (training): %f" % (z1LossTrainAvg[-1])
print "Square loss (test): %f" % (squareLossTestAvg[-1])
print "0/1 loss (test): %f" % (z1LossTestAvg[-1])