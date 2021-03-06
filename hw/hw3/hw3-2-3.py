# CSE 546 Homework 3
# Problem 2: MNIST revisited - Extra credit

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
# #											Part 2.3 - Extra Credit: Neural net features
# #-------------------------------------------------------------------------------------------------------------


# 
# Define some functions that we will need
# 

# ---Random Neural Net Feature generator
# 
# Generates random Neural net features
def NNFeatVec(x,V):
	Vx = V.dot(x)
	Vx[Vx < 0] = 0
	return Vx

# ---Get all probabilities---
# 
# Gets softmax probabilites P(y=l|x,w) for all (x,y)
# P(i,j) = P(y^j=i+1|x^i,w)
def getAProbs(X,W,V):
	k = W.shape[0]+1
	probs = np.empty((k,X.shape[0]))

	# Precompute some quantities
	WX = np.empty((W.shape[0],X.shape[0]))
	for j in range(X.shape[0]):
		WX[:,j] = W.dot(NNFeatVec(X[j,:],V))

	ex = np.exp(WX)
	denom = -np.log(1 + np.sum(ex,0))

	# Compute all the probabilities
	probs[0,:] = np.exp(denom)
	for j in range(1,k):
		probs[j,:] = np.exp(WX[j-1,:] + denom)
	return probs


# ---Stochastic Gradient---
# 
# Computes an approximate gradient of the log loss function using sample points specified
# by sampleIndices
def stochGradient(Y,X,V,W,reg,sampleIndices):
	k = X.shape[1]
	nClasses = W.shape[0]+1
	output = np.zeros(W.shape)

	P = getAProbs(X[sampleIndices,:],W,V)[1:,:]
	rbf = NNFeatVec(X[sampleIndices,:].T,V).T

	for ii in range(nClasses-1):
		output[ii,:] = -rbf.T.dot((Y[sampleIndices]==ii+1).astype(int) - P[ii,:])
	return (output / batchSize) + reg * W


# ---0/1 loss---
# 
# Computes the 0/1 loss
def z1Loss(Y,X,V,W):
	P = getAProbs(X,W,V)
	model_out = np.argmax(P,0)
	return (1. * np.count_nonzero(model_out - Y)) / len(Y)


# ---Log loss---
# 
# Log loss for this problem
def logLoss(Y,X,V,W,reg):
	P = getAProbs(X,W,V)
	total = np.sum(np.log(P[Y,range(len(Y))]))
	return -(total + reg  * np.sum(W**2) / 2.) / len(Y)


# **Change**
# ---0/1 and log loss---
# 
# Gets the 0/1 and log losses simultaneously so we don't have to recompute feature vectors
# Doe this for both weights and average weights over last epoch
def getLoss(Y,X,V,W,WAvg,reg):

	k = W.shape[0]+1
	P = np.empty((k,X.shape[0]))
	PAvg = np.empty((k,X.shape[0]))

	# Precompute some quantities
	WX = np.empty((W.shape[0],X.shape[0]))
	WAvgX = np.empty((WAvg.shape[0],X.shape[0]))
	for j in range(X.shape[0]):
		rbf = NNFeatVec(X[j,:],V)
		WX[:,j] = W.dot(rbf)
		WAvgX[:,j] = WAvg.dot(rbf)

	ex = np.exp(WX)
	exAvg = np.exp(WAvgX)
	denom = -np.log(1 + np.sum(ex,0))
	denomAvg = -np.log(1 + np.sum(exAvg,0))

	# Compute all the probabilities
	P[0,:] = np.exp(denom)
	PAvg[0,:] = np.exp(denomAvg)
	for j in range(1,k):
		P[j,:] = np.exp(WX[j-1,:] + denom)
		PAvg[j,:] = np.exp(WAvgX[j-1,:]+denomAvg)

	# Compute 0/1 loss
	model_out = np.argmax(P,0)
	model_outAvg = np.argmax(PAvg,0)
	z1_out = (1. * np.count_nonzero(model_out - Y)) / len(Y)
	z1_outAvg = (1. * np.count_nonzero(model_outAvg - Y)) / len(Y)

	# Compute log loss
	total = np.sum(np.log(P[Y,range(len(Y))]))
	totalAvg = np.sum(np.log(PAvg[Y,range(len(Y))]))
	log_out = -(total + reg  * np.sum(W**2) / 2.) / len(Y)
	log_outAvg = -(totalAvg + reg  * np.sum(WAvg**2) / 2.) / len(Y)

	return (log_out,z1_out,log_outAvg,z1_outAvg)



# ---Stochastic gradient descent---
# 
# Stochastic gradient descent method (log loss)
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
# 
def SGD(YTrain,YTest,XTrain,XTest,V,reg,nClasses,batchSize=100,numEpochs=10,TOL=1.e-5,w=None):
	t3 = time.time()
	if w is None:
		w = np.zeros((nClasses,V.shape[0]))		# Note: this changed from before

	# wOld = np.zeros(w.shape)
	N = XTrain.shape[0]
	num = 1.e-8
	step = num
	wAvg = np.copy(w)

	logLossTrain = np.zeros(numEpochs + 1)
	logLossTest = np.zeros(numEpochs + 1)
	z1LossTrain = np.zeros(numEpochs + 1)
	z1LossTest = np.zeros(numEpochs + 1)

	logLossTrainAvg = np.zeros(numEpochs + 1)
	logLossTestAvg = np.zeros(numEpochs + 1)
	z1LossTrainAvg = np.zeros(numEpochs + 1)
	z1LossTestAvg = np.zeros(numEpochs + 1)

	# Random subsets of 10000 points at which to evaluate log loss
	randTrainSet = np.random.choice(np.arange(YTrain.shape[0]),10000,replace=False)

	totalSteps = 0
	# Loop over epochs
	for it in range(0,numEpochs):

		t5 = time.time()
		# Check log loss and 0/1 loss every time we pass through all the points
		(logLossTrain[it],z1LossTrain[it],logLossTrainAvg[it],z1LossTrainAvg[it]) = getLoss(YTrain[randTrainSet],XTrain[randTrainSet,:],V,w,wAvg,reg)
		(logLossTest[it],z1LossTest[it],logLossTestAvg[it],z1LossTestAvg[it]) = getLoss(YTest,XTest,V,w,wAvg,reg)

		t6 = time.time()
		print "Time to compute the log loss: %f"%(t6-t5)
		print "Log loss at iteration %d (w): %f"%(it,logLossTrain[it])
		print "0/1 loss at iteration %d (w): %f"%(it,z1LossTrain[it])
		# print "Log loss at iteration %d (wAvg): %f"%(it,logLossTrainAvg[it])
		# print "0/1 loss at iteration %d (wAvg): %f"%(it,z1LossTrainAvg[it])

		# Compute new order in which to visit indices
		sampleIndices = np.random.permutation(N)

		# Reset wAvg for next epoch
		wAvg = np.copy(w)

		# Loop over all the points
		for subIt in range(0,N / batchSize):
			# Compute the gradient
			currentIndices = sampleIndices[subIt*batchSize:(subIt+1)*batchSize]
			newGrad = stochGradient(YTrain,XTrain,V,w,reg,currentIndices)

			# Precompute a possibly expensive quantity
			gradNorm = np.sqrt(np.sum(newGrad**2))
			# print "Order of gradient: %f\n" %np.log10(gradNorm)

			# Take a step in the negative gradient direction
			step = num / np.sqrt(subIt+1)
			w = w - step * newGrad
			wAvg += w

			if gradNorm < TOL:
				# Method has converged, so record loss and exit
				(logLossTrain[it],z1LossTrain[it],logLossTrainAvg[it],z1LossTrainAvg[it]) = getLoss(YTrain[randTrainSet],XTrain[randTrainSet],V,w,wAvg,reg)
				(logLossTest[it],z1LossTest[it],logLossTestAvg[it],z1LossTestAvg[it]) = getLoss(YTest,XTest,V,w,wAvg,reg)

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

	(logLossTrain[it+1],z1LossTrain[it+1],logLossTrainAvg[it+1],z1LossTrainAvg[it+1]) = getLoss(YTrain,XTrain,V,w,wAvg,reg)
	(logLossTest[it+1],z1LossTest[it+1],logLossTestAvg[it+1],z1LossTestAvg[it+1]) = getLoss(YTest,XTest,V,w,wAvg,reg)

	t4 = time.time()
	print "Time to perform %d epochs of SGD with batch size %d: %f"%(it+1,batchSize,t4-t3)
	
	return (w,wAvg,logLossTrain[:(it+2)],z1LossTrain[:(it+2)],logLossTest[:(it+2)],z1LossTest[:(it+2)],logLossTrainAvg[:(it+2)],z1LossTrainAvg[:(it+2)],logLossTestAvg[:(it+2)],z1LossTestAvg[:(it+2)])


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

# Generate or load random vectors for features
if os.path.isfile("feats.npy"):
	feats = np.load("feats.npy")
else:
	feats = np.random.randn(trainProj.shape[0],trainProj.shape[1])
	np.save("feats",feats)
 
# Set parameters
# --------------------------------------------------------------------------
nClasses = 10		# Number of classes
reg = 1.			# Regularization parameter
batchSize = 10		# Batch size for SGD
numEpochs = 20		# Number of epochs to go through before terminating
# --------------------------------------------------------------------------

print "Time elapsed during setup: %f" %(time.time() - t1)


# Run the method
(w, wAvg, logLossTrain, z1LossTrain,
logLossTest, z1LossTest,
logLossTrainAvg, z1LossTrainAvg,
logLossTestAvg, z1LossTestAvg) = SGD(train_label,
										test_label,
										trainProj,
										testProj,
										feats,
										reg,
										nClasses,
										batchSize,
										numEpochs)


# Plot log loss on test and train sets
matplotlib.rcParams.update({'font.size' : 20})
n = len(logLossTrain)-1
outputCond = train_label.shape[0] / batchSize
its = range(0,n*outputCond+1,outputCond)
plt.figure(1)
plt.plot(its,logLossTrain,'b-o',its,logLossTrainAvg,'k--o',
	its,logLossTest,'r-x',its,logLossTestAvg,'g--x',linewidth=1.5)
plt.xlabel('Iteration')
plt.ylabel('Log loss')
plt.legend(['Training (w)','Training (wAvg)','Test (w)', 'Test (wAvg)'])
# plt.title('Log loss')

# Plot 0/1 loss on test and train sets
plt.figure(2)
plt.plot(its[1:],z1LossTrain[1:],'b-o',its[1:],z1LossTrainAvg[1:],'k--o',
	its[1:],z1LossTest[1:],'r-x',its[1:],z1LossTestAvg[1:],'g--x',linewidth=1.5)
plt.xlabel('Iteration')
plt.ylabel('0/1 loss')
plt.legend(['Training (w)','Training (wAvg)','Test (w)', 'Test (wAvg)'])
# plt.title('0/1 loss')
plt.show()

# Print out final 0/1 and log loss
print "Log loss (training): %f" % (logLossTrain[-1])
print "0/1 loss (training): %f" % (z1LossTrain[-1])
print "Log loss (test): %f" % (logLossTest[-1])
print "0/1 loss (test): %f" % (z1LossTest[-1])
