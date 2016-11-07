# CSE 546 Homework 2
# Problem 2.2: Softmax classification--gradient descent

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

#-------------------------------------------------------------------------------------------------------------
#									Part 2.2 - Softmax classification with GD
#-------------------------------------------------------------------------------------------------------------

# Gets P(y=l|x,w) for each class l=0,1,...,k-1
# Note: W = [--w1--
			#--w2--
			#  .
			#  .
			#--wkm1]
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

# Log loss for this problem
def logLoss(Y,X,W,reg):
	total = 0.0
	P = getAProbs(X,W)
	total = np.sum(np.log(P[Y,range(len(Y))]))
	return -(total + reg  * np.linalg.norm(W,'fro')**2 / 2.) / len(Y)

# Computes the full gradient of the log likelihood function
def Gradient(Y,X,W,reg):
	N = X.shape[0]
	k = X.shape[1]
	nClasses = W.shape[0]+1
	output = np.zeros(W.shape)
	P = getAProbs(X,W)[1:,:]

	for ii in range(nClasses-1):
		output[ii,:] = -np.dot(X.T,(Y==ii+1).astype(int) - P[ii,:])
	return (output) / N + reg * W


# Computes the 0/1 loss
def z1Loss(Y,X,w):
	P = getAProbs(X,w)
	model_out = np.argmax(P,0)
	return (1.0 * np.count_nonzero(model_out - Y)) / len(Y)


# Use Gradient Descent to miminimize log loss
def GD(Ytrain,Ytest,Xtrain,Xtest,reg,nClasses,w=None,TOL=1.e-2,MAX_ITS=1000):
	if w is None:
		w = np.zeros(((nClasses-1),np.shape(Xtrain)[1]))

	N = Xtrain.shape[0]
	num = 1.e-4 / 8.0
	step = num

	logLossTrain = np.empty(MAX_ITS)
	logLossTest = np.empty(MAX_ITS)
	z1LossTrain = np.empty(MAX_ITS)
	z1LossTest = np.empty(MAX_ITS)

	for it in range(0,MAX_ITS):
		newGrad = Gradient(Ytrain,Xtrain,w,reg)

		# Precompute a possibly expensive quantity
		gradNorm = np.linalg.norm(newGrad,'fro')

		logLossTrain[it] = logLoss(Ytrain,Xtrain,w,reg)
		logLossTest[it] = logLoss(Ytest,Xtest,w,reg)
		z1LossTrain[it] = z1Loss(Ytrain,Xtrain,w)
		z1LossTest[it] = z1Loss(Ytest,Xtest,w)

		# Take a step in the negative gradient direction
		w = w - step * newGrad

		# print "Time elapsed evaluating log loss: %f"%(t5-t4)
		print "Log loss at iteration %d: %f"%(it,logLossTrain[it])
		print "0/1 loss at iteration %d: %f"%(it,z1LossTrain[it])
		print "Order of gradient: %f\n" %np.log10(gradNorm)

		if gradNorm < TOL:
			break

	if (it == MAX_ITS-1):
		print "Warning: Maximum number of iterations reached."

	return (w,logLossTrain[:(it+1)],z1LossTrain[:(it+1)],logLossTest[:(it+1)],z1LossTest[:(it+1)])

# ------------------------------------------------------------------------------------------
# Set some parameters
nClasses = 10		# Number of classes
reg = 1.0			# Regularization parameter

# Run the method
(w,logLossTrain,z1LossTrain,logLossTest,z1LossTest) = GD(train_label,test_label,train_img,test_img,reg,nClasses)

# Plot log loss on test and train sets
plt.figure(1)
plt.plot(range(0,len(logLossTrain)),logLossTrain,'b-',range(0,len(logLossTrain)),logLossTest,'r-')
plt.xlabel('Iteration')
plt.ylabel('Log loss')
plt.legend(['Training','Test'])
# plt.title('Log loss for multi-class logistic regression (batch gradient descent)')
plt.show()

# Plot 0/1 loss on the test and train sets
plt.figure(2)
plt.plot(range(0,len(z1LossTrain)),z1LossTrain,'b-',range(0,len(logLossTrain)),z1LossTest,'r-')
plt.xlabel('Iteration')
plt.ylabel('0/1 loss')
plt.legend(['Training','Test'])
# plt.title('0/1 loss for multi-class logistic regression (batch gradient descent)')
plt.show()

# Print out final 0/1 and log loss
print "log loss (training): %f" % (logLossTrain[-1])
print "0/1 loss (training): %f" % (z1LossTrain[-1])
print "log loss (test): %f" % (logLossTest[-1])
print "0/1 loss (test): %f" % (z1LossTest[-1])