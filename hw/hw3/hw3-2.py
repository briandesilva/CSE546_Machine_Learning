# CSE 546 Homework 3
# Problem 2: MNIST revisited

# Brian de Silva
# 1422824

import numpy as np
from mnist import MNIST
import os
import time
import matplotlib.pyplot as plt

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
def RBFVec(x,X,sigma):
	return np.exp(-(np.linalg.norm(X-x,2,1)**2) / (2. * sigma ** 2))

# ---Stochastic Gradient---
# 
# Computes an approximate gradient of the square loss function using sample points specified
# by sampleIndices
def stochGradient(Y,X,W,reg,sampleIndices,sigma):
	output = np.zeros(np.prod(W.shape))
	labels = np.arange(0,10)
	for ii in range(0,sampleIndices.shape[0]):
		output += 2. * np.kron(W.dot(RBFVec(X[sampleIndices[ii],:],X,sigma)) - (Y[sampleIndices[ii]]==labels).astype(int),RBFVec(X[sampleIndices[ii],:],X,sigma))
	return output.reshape(W.shape) + reg * W

# ---0/1 loss---
# 
# Computes the 0/1 loss
def z1Loss(Y,X,w,sigma):
	# Get predictions for each point
	model_out = np.empty(Y.shape[0])
	for k in range(0,Y.shape[0]):
		model_out[k] = np.argmax(w.dot(RBFVec(X[k,:],X,sigma)))
	return (1.0 * np.count_nonzero(model_out - Y)) / Y.shape[0]

# ---Square loss---
# 
# square loss for this problem
def squareLoss(Y,X,W,reg,sigma):
	total = 0.0
	for k in range(0,Y.shape[0]):
		total += np.linalg.norm((np.arange(0,10)==Y[k]).astype(int) - W.dot(RBFVec(X[k,:],X,sigma))) ** 2
	return total + reg * np.linalg.norm(W,'fro') ** 2 / 2.


# times = np.empty(15)
# numPts = np.empty(15)
# numPts[0:5] = np.arange(1,6)
# numPts[5:] = np.arange(10,101,10)
# it = 0
# for k in numPts:
# 	t = time.time()
# 	stochGradient(train_label,trainProj,W,0.0,sampleIndices[:k],sigma)
# 	times[it] = time.time() - t
# 	it += 1



t = time.time()
z = z1Loss(train_label,trainProj,W,sigma)
print "Elapsed time: %f"%(time.time()-t)


# g = stochGradient(train_label,trainProj,W,0.0,sampleIndices[:1],sigma)

# w = W.dot(RBFVec(trainProj[10,:],trainProj,sigma))

# # ---Stochastic gradient descent---
# # 
# # Stochastic gradient descent method (square loss)
# def SGD(Ytrain,Ytest,Xtrain,Xtest,reg,nClasses,sigma,batchSize=100,numEpochs=10,TOL=1.e-5,w=None):
# 	t3 = time.time()
# 	if w is None:
# 		w = np.zeros((nClasses,np.shape(Ytrain)[0]))	# Note: this changed from before

# 	# wOld = np.zeros(w.shape)
# 	N = Xtrain.shape[0]
# 	num = 1.e-5
# 	step = num
# 	outputCond = 15000 / batchSize
# 	wAvg = np.zeros(w.shape)

# 	squareLossTrain = np.empty(numEpochs + 1)
# 	squareLossTest = np.empty(numEpochs + 1)
# 	z1LossTrain = np.empty(numEpochs + 1)
# 	z1LossTest = np.empty(numEpochs + 1)

# 	squareLossTrainAvg = np.empty(numEpochs + 1)
# 	squareLossTestAvg = np.empty(numEpochs + 1)
# 	z1LossTrainAvg = np.empty(numEpochs + 1)
# 	z1LossTestAvg = np.empty(numEpochs + 1)	


# 	# Loop over epochs
# 	for it in range(0,numEpochs):

# 		t5 = time.time()
# 		# Check square loss and 0/1 loss every time we pass through all the points
# 		squareLossTrain[it] = squareLoss(Ytrain,Xtrain,w,reg,sigma)
# 		# squareLossTest[it] = squareLoss(Ytest,Xtest,w,reg,sigma)
# 		# z1LossTrain[it] = z1Loss(Ytrain,Xtrain,w,sigma)
# 		# z1LossTest[it] = z1Loss(Ytest,Xtest,w,sigma)
# 		t6 = time.time()
# 		print "Time to compute the square loss: %f"%(t5-t6)

# 		squareLossTrainAvg[it] = squareLoss(Ytrain,Xtrain,wAvg,reg,sigma)
# 		# squareLossTestAvg[it] = squareLoss(Ytest,Xtest,wAvg,reg,sigma)
# 		# z1LossTrainAvg[it] = z1Loss(Ytrain,Xtrain,wAvg,sigma)
# 		# z1LossTestAvg[it] = z1Loss(Ytest,Xtest,wAvg,sigma)


# 		print "Square loss at iteration (w) %d: %f"%(it,squareLossTrain[it])
# 		# print "0/1 loss at iteration (w) %d: %f"%(it,z1LossTrain[it])
# 		print "Square loss at iteration (wAvg) %d: %f"%(it,squareLossTrainAvg[it])
# 		# print "0/1 loss at iteration (wAvg) %d: %f"%(it,z1LossTrainAvg[it])

# 		# Compute new order in which to visit indices
# 		sampleIndices = np.random.permutation(N)

# 		# Zero out wAvg for next epoch
# 		wAvg = np.zeros(w.shape)

# 		# Loop over all the points
# 		for subIt in range(0,N / batchSize):
# 			# Compute the gradient
# 			currentIndices = sampleIndices[subIt*batchSize:(subIt+1)*batchSize]
# 			t7 = time.time()
# 			newGrad = stochGradient(Ytrain,Xtrain,w,reg,currentIndices,sigma)
# 			t8 = time.time()
# 			print "Time to compute the gradient: %f"%(t7-t8)

# 			# Precompute a possibly expensive quantity
# 			gradNorm = np.linalg.norm(newGrad,'fro')

# 			# Take a step in the negative gradient direction
# 			step = num / np.sqrt(it+1)
# 			w = w - step * newGrad
# 			wAvg += w

# 			if gradNorm < TOL:
# 				squareLossTrain[it] = squareLoss(Ytrain,Xtrain,w,reg,sigma)
# 				squareLossTest[it] = squareLoss(Ytest,Xtest,w,reg,sigma)
# 				z1LossTrain[it] = z1Loss(Ytrain,Xtrain,w,sigma)
# 				z1LossTest[it] = z1Loss(Ytest,Xtest,w,sigma)

# 				squareLossTrainAvg[it] = squareLoss(Ytrain,Xtrain,wAvg,reg,sigma)
# 				squareLossTestAvg[it] = squareLoss(Ytest,Xtest,wAvg,reg,sigma)
# 				z1LossTrainAvg[it] = z1Loss(Ytrain,Xtrain,wAvg,sigma)
# 				z1LossTestAvg[it] = z1Loss(Ytest,Xtest,wAvg,sigma)
# 				print "Order of gradient: %f\n" %np.log10(gradNorm)
# 				wAvg /= (subIt + 1)
# 				break

# 		# Compute average weight over previous epoch
# 		wAvg /= (N / batchSize)

# 		# Print out size of the gradient
# 		print "Order of gradient: %f\n" %np.log10(gradNorm)

# 	if (it == MAX_ITS-1):
# 		print "Warning: Maximum number of iterations reached."

# 	squareLossTrain[it+1] = squareLoss(Ytrain,Xtrain,w,reg,sigma) 
# 	# squareLossTest[it+1] = squareLoss(Ytest,Xtest,w,reg,sigma)
# 	# z1LossTrain[it+1] = z1Loss(Ytrain,Xtrain,w,sigma)
# 	# z1LossTest[it+1] = z1Loss(Ytest,Xtest,w,sigma)

# 	squareLossTrainAvg[it+1] = squareLoss(Ytrain,Xtrain,wAvg,reg,sigma)
# 	# squareLossTestAvg[it+1] = squareLoss(Ytest,Xtest,wAvg,reg,sigma)
# 	# z1LossTrainAvg[it+1] = z1Loss(Ytrain,Xtrain,wAvg,sigma)
# 	# z1LossTestAvg[it+1] = z1Loss(Ytest,Xtest,wAvg,sigma)

# 	t4 = time.time()
# 	print "Time to perform %d iterations of SGD with batch size %d: %f"%(it+1,batchSize,t4-t3)
# 	return (w,squareLossTrain[:(it+2)],z1LossTrain[:(it+2)],squareLossTest[:(it+2)],z1LossTest[:(it+2)],squareLossTrainAvg[:(it+2)],z1LossTrainAvg[:(it+2)],squareLossTestAvg[:(it+2)],z1LossTestAvg[:(it+2)])



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
	inds = np.random.choice(np.arange(0,N),2)	# Get a random pair of data points
	dists += np.linalg.norm(trainProj[inds[0],:] - trainProj[inds[1],:] ,2)

# Cheat a little and use the empirical mean distance between points
dists /= numSamples

# Set parameters
sigma = dists / 2.
nClasses = 10		# Number of classes
reg = 0.0			# Regularization parameter
batchSize = 100		# Batch size for SGD
numEpochs = 10		# Number of epochs to go through before terminating

print "Time elapsed during setup: %f" %(time.time() - t1)


# Run the method
(w, squareLossTrain, z1LossTrain,
squareLossTest, z1LossTest,
squareLossTrainAvg, z1LossTrainAvg,
squareLossTestAvg, z1LossTestAvg) = SGD(train_label,
										test_label,
										trainProj,
										testProj,
										reg,
										nClasses,
										sigma,
										batchSize,
										numEpochs)


# Plot square loss on test and train sets
n = len(squareLossTrain)-1
outputCond = train_label.shape[0] / batchSize
its = range(0,n*outputCond+1,outputCond)
plt.figure(1)
plt.plot(its,squareLossTrain,'b-o',its,squareLossTrainAvg,'k-o',
	its,squareLossTest,'r-x',its,squareLossTestAvg,'g-x')
plt.xlabel('Iteration')
plt.ylabel('Square loss')
plt.legend(['Training (w)','Training (wAvg)','Test (w)', 'Test (wAvg)'])
# plt.title('Square loss')
plt.show()

# Plot 0/1 loss on test and train sets
plt.figure(2)
plt.plot(its,z1LossTrain,'b-o',its,z1LossTrainAvg,'k-o',
	its,z1LossTest,'r-x',its,z1LossTestAvg,'g-x')
plt.xlabel('Iteration')
plt.ylabel('0/1 loss')
plt.legend(['Training (w)','Training (wAvg)','Test (w)', 'Test (wAvg)'])
# plt.title('0/1 loss')
plt.show()

# Print out final 0/1 and square loss
print "Square loss (training): %f" % (squareLossTrainAvg[-1])
print "0/1 loss (training): %f" % (z1LossTrainAvg[-1])
print "Square loss (test): %f" % (squareLossTestAvg[-1])
print "0/1 loss (test): %f" % (z1LossTestAvg[-1])







# ***Might need this later***
# # ---Stochastic Gradient---
# # 
# # Computes an approximate gradient of the log likelihood function using sample points specified
# # by sampleIndices
# def stochGradient(Y,X,W,reg,sampleIndices):
# 	N = X.shape[0]
# 	k = X.shape[1]
# 	nClasses = W.shape[0]+1
# 	output = np.zeros(W.shape)

# 	# sampleIndices = np.random.choice(np.arange(0,N),batchSize)
# 	P = getAProbs(X[sampleIndices,:],W)[1:,:]

# 	for ii in range(nClasses-1):
# 		output[ii,:] = -X[sampleIndices,:].T.dot((Y[sampleIndices]==ii+1).astype(int) - P[ii,:])
# 	return (output / sampleIndices.shape[0]) + reg * W