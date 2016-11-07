# CSE 546 Homework 2
# Problem 2.4: Softmax classification--Neural nets with a random first layer 

# Brian de Silva
# 1422824

import numpy as np
# from scipy.special import expit		# Sigmoid function
from mnist import MNIST
import os
import time
import matplotlib.pyplot as plt

t0 = time.time()

# Import the data
mndata = MNIST('../hw1/mnist')
train_img, train_label = mndata.load_training()
test_img, test_label = mndata.load_testing()

t1 = time.time()
print "Time elapsed loading data: %f" %(t1-t0)

train_label = np.array(train_label)
test_label = np.array(test_label)
train_img = np.array(train_img)
test_img = np.array(test_img)

# Create binary labels
train_bin_label = np.zeros(len(train_label))
test_bin_label = np.zeros(len(test_label))
train_bin_label[train_label == 2] = 1
test_bin_label[test_label == 2] = 1

t2 = time.time()
print "Time elapsed creating labels: %f" %(t2-t1)


# #-------------------------------------------------------------------------------------------------------------
# #						Part 2.4 - Softmax classification with SGD - Using more features
# #-------------------------------------------------------------------------------------------------------------

# Gets probabilites P(y=l|x,w) for all (x,y)
# P(i,j) = P(y^j=i+1|x^i,w)
def getAProbs(X,W):
	k = W.shape[0]+1
	probs = np.empty((k,X.shape[0]))
	# Precompute some quantities
	WX = np.dot(W,X.T)
	ex = np.exp(WX)
	denom = -np.log(1 + np.sum(ex,0))
	# Compute all the probabilities
	probs[0,:] = np.exp(denom)
	for j in range(1,k):
		probs[j,:] = np.exp(WX[j-1,:] + denom)
	return probs

# Gets P(y=l|x,w) for a single class l
def getProb(y,x,W):
	Wx = np.dot(W,x)
	
	denom = -np.log(1 + np.sum(np.exp(Wx)))
	if y == 0:
		return np.exp(denom)
	else:
		return np.exp(Wx[y-1] + denom)

def getProbs(x,W):
	k = W.shape[0]+1
	probs = np.empty(k)

	# Precompute some quantities
	Wx = np.dot(W,x)
	ex = np.exp(Wx)
	denom = -np.log(1 + np.sum(ex))

	# Compute all the probabilities
	probs[0] = np.exp(denom)
	for j in range(1,k):
		probs[j] = np.exp(Wx[j-1] + denom)
	return probs

# Computes an approximate gradient of the log likelihood function using batchSize sample points
def stochGradient(Y,X,W,reg,batchSize):
	N = X.shape[0]
	k = X.shape[1]
	nClasses = W.shape[0]+1
	output = np.zeros(W.shape)

	sampleIndices = np.random.choice(np.arange(0,N),batchSize)
	P = getAProbs(X[sampleIndices,:],W)[1:,:]

	for ii in range(nClasses-1):
		output[ii,:] = -X[sampleIndices,:].T.dot((Y[sampleIndices]==ii+1).astype(int) - P[ii,:])
	return (output / batchSize) + reg * W

# Computes the 0/1 loss
def z1Loss(Y,X,w):
	P = getAProbs(X,w)
	model_out = np.argmax(P,0)
	return (1.0 * np.count_nonzero(model_out - Y)) / len(Y)

# Log loss for this problem
def logLoss(Y,X,W,reg):
	total = 0.0
	P = getAProbs(X,W)
	total = np.sum(np.log(P[Y,range(len(Y))]))
	return -(total + reg  * np.linalg.norm(W,'fro')**2 / 2.) / len(Y)

# Stochastic gradient descent
# Similar to GD, except uses stochGradient function instead of Gradient and doesn't compute loss at every iteration
def SGD(Ytrain,Ytest,Xtrain,Xtest,reg,nClasses,batchSize=1,w=None,MAX_ITS=150*100+1,TOL=1.e-5):
	t3 = time.time()
	if w is None:
		w = np.zeros(((nClasses-1),np.shape(Xtrain)[1]))

	# wOld = np.zeros(w.shape)
	N = Xtrain.shape[0]
	num = 1.e-7
	step = num
	outputCond = 15000 / batchSize

	logLossTrain = np.empty(MAX_ITS / outputCond + 1)
	logLossTest = np.empty(MAX_ITS / outputCond + 1)
	z1LossTrain = np.empty(MAX_ITS / outputCond + 1)
	z1LossTest = np.empty(MAX_ITS / outputCond + 1)

	for it in range(0,MAX_ITS):
		newGrad = stochGradient(Ytrain,Xtrain,w,reg,batchSize)
		

		# Precompute a possibly expensive quantity
		gradNorm = np.linalg.norm(newGrad,'fro')

		# Check Log loss and 0/1 loss every 15000 points we look through
		if np.mod(it,outputCond) == 0:
			logLossTrain[it / outputCond] = logLoss(Ytrain,Xtrain,w,reg)
			logLossTest[it / outputCond] = logLoss(Ytest,Xtest,w,reg)
			z1LossTrain[it / outputCond] = z1Loss(Ytrain,Xtrain,w)
			z1LossTest[it / outputCond] = z1Loss(Ytest,Xtest,w)

			print "Log loss at iteration %d: %f"%(it,logLossTrain[it / outputCond])
			print "0/1 loss at iteration %d: %f"%(it,z1LossTrain[it / outputCond])
			print "Order of gradient: %f\n" %np.log10(gradNorm)

		# Take a step in the negative gradient direction
		step = num / np.sqrt(it+1)
		w = w - step * newGrad

		if gradNorm < TOL:
			logLossTrain[it / outputCond + 1] = logLoss(Ytrain,Xtrain,w,reg)
			logLossTest[it / outputCond + 1] = logLoss(Ytest,Xtest,w,reg)
			z1LossTrain[it / outputCond + 1] = z1Loss(Ytrain,Xtrain,w)
			z1LossTest[it / outputCond + 1] = z1Loss(Ytest,Xtest,w)
			print "Order of gradient: %f\n" %np.log10(gradNorm)
			break

	if (it == MAX_ITS-1):
		print "Warning: Maximum number of iterations reached."

	idx = it / outputCond + 2
	t4 = time.time()
	print "Time to perform %d iterations of SGD with batch size %d: %f"%(it+1,batchSize,t4-t3)
	return (w,logLossTrain[:idx],z1LossTrain[:idx],logLossTest[:idx],z1LossTest[:idx])


# ----------------------------------------------------------------

# # Parameters used:
# # step = 1.e-7 / (4.0*sqrt(it+1))
# # batch size = 100
# # reg = 1.0

# Compute new features
V = np.load('features.npy')
train_img = np.dot(train_img,V)
train_img[train_img < 0] = 0
test_img = np.dot(test_img,V)
test_img[test_img < 0] = 0

# Add column of all 1's to simulate offset w0
# train_img = np.concatenate((np.ones((train_img.shape[0],1)),train_img),axis=1)
# test_img = np.concatenate((np.ones((test_img.shape[0],1)),test_img),axis=1)

# Set some parameters
nClasses = 10		# Number of classes
reg = 1.0			# Regularization parameter
batchSize = 100		# Number of points to sample to approximate gradient

# Run the method
(w,logLossTrain,z1LossTrain,logLossTest,z1LossTest) = SGD(train_label,test_label,train_img,test_img,reg,nClasses,batchSize)

# Plot log loss on test and train sets
n = len(logLossTrain)-1
outputCond = 15000 / batchSize
its = range(0,n*outputCond+1,outputCond)
plt.figure(1)
plt.plot(its,logLossTrain,'b-o',its,logLossTest,'r-x')
plt.xlabel('Iteration')
plt.ylabel('Log loss')
plt.legend(['Training','Test'])
plt.show()

# Plot 0/1 loss on the test and train sets
plt.figure(2)
plt.plot(its,z1LossTrain,'b-o',its,z1LossTest,'r-x')
plt.xlabel('Iteration')
plt.ylabel('0/1 loss')
plt.legend(['Training','Test'])
plt.show()

# Print out final 0/1 and log loss
print "log loss (training): %f" % (logLossTrain[-1])
print "0/1 loss (training): %f" % (z1LossTrain[-1])
print "log loss (test): %f" % (logLossTest[-1])
print "0/1 loss (test): %f" % (z1LossTest[-1])