# CSE 546 Homework 2 -- Problem 2.1: Multi-class Classification using Logistic
# Regression and Softmax

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

t2 = time.time()
print "Time elapsed creating labels: %f" %(t2-t1)

#-------------------------------------------------------------------------------------------------------------
#									Part 2.1 - Binary Logistic Regression
#-------------------------------------------------------------------------------------------------------------


# # Weight vector
# w = np.zeros(np.shape(train_img)[1])
# w0 = 0.0

# # Regularization parameter
# reg = 1.0

# # Log loss (function we're trying to minimize)
# def logLoss(Y,X,w,w0,reg):
# 	Xw = np.dot(X,w)
# 	c1 = np.dot(Y.T,Xw + w0)
# 	c2 = np.sum(np.log(1 + np.exp(w0 + Xw)))
# 	c3 = (reg / 2.0) * np.linalg.norm(w)**2
# 	return (c2 + c3 - c1)/len(Y)

# # Computes the full gradient of the log likelihood function
# def fullGradient(Y,X,w,w0,reg):
# 	fullGrad = np.empty(len(w)+1)
# 	C1 = np.exp(w0 + np.dot(X,w))
# 	C2 = Y - (C1 / (1.0 + C1))
# 	fullGrad[0] = np.sum(C2)
# 	fullGrad[1:] = (np.dot(X.T,C2) - reg * w) 
# 	return fullGrad / len(Y)

# Use Gradient Descent to miminimize log loss
# def GD(Ytrain,X,Ytest,Xtest,reg,w=None,w0=0.,TOL=1.e-2,MAX_ITS=500):
# 	if w is None:
# 		w = np.zeros(np.shape(X)[1])

# 	N = len(X)
# 	# num = (1.e-4)/3.0
# 	num = 1.e-5
# 	step = num
# 	# step = num/N
# 	# prevGrad = np.ones(len(w)+1)
# 	lossVecTrain = np.empty(MAX_ITS)
# 	lossVecTest = np.empty(MAX_ITS)
# 	# step = 1.e-9

# 	for it in range(0,MAX_ITS):
# 		# t3 = time.time()
# 		newGrad = fullGradient(Ytrain,X,w,w0,reg)
# 		# t4 = time.time()

# 		# print "Time elapsed computing gradient (iteration %d): %f"%(it,t4-t3)

# 		# w = w + (step / np.sqrt(it+1)) * newGrad[1:]
# 		# w0 = w0 + (step / np.sqrt(it+1)) * newGrad[0]

# 		# # Backtracking line search
# 		# # step = 10.0
# 		# rho = 1./2.

# 		# # Precompute some expensive quantities
# 		# gradNorm = np.linalg.norm(newGrad)
# 		# lgLoss = logLoss(Ytrain,X,w,w0,reg)
# 		# count = 1

# 		# while logLoss(Ytrain,X,w + step * newGrad[1:], w0 + step * newGrad[0],reg) > lgLoss - (2./3.)*step * gradNorm:
# 		# 	step = rho * step
# 		# 	count += 1
# 		# print "Number of steps tested: %d" %count
# 		w = w + step * newGrad[1:]
# 		w0 = w0 + step * newGrad[0]
# 		lossVecTrain[it] = logLoss(Ytrain,X,w,w0,reg)
# 		# print "Step size: %.12f" %step

# 		# See if we can increase step a bit
# 		# step = step * 1.2

# 		lossVecTest[it] = logLoss(Ytest,Xtest,w,w0,reg)
# 		# t5 = time.time()

# 		# print "Time elapsed evaluating log loss: %f"%(t5-t4)
# 		print "Log loss at iteration %d: %f"%(it,lossVecTrain[it])
# 		print "Order of gradient: %f\n" %np.log10(np.linalg.norm(newGrad,np.inf))

# 		if np.linalg.norm(newGrad,np.inf) < TOL:
# 			break
# 		# prevGrad = np.copy(newGrad)

# 	if (it == MAX_ITS-1):
# 		print "Warning: Maximum number of iterations reached."
# 	# print "Step size used: %f / %d = %f"%(num,N,step)

# 	# return (w,w0,lossVecTrain[:it])
# 	return (w,w0,lossVecTrain[:it],lossVecTest[:it])

# (w,w0,lossVecTrain,lossVecTest) = GD(train_bin_label,train_img,test_bin_label,test_img,reg)

# # Classify the points
# train_out = w0 + np.dot(train_img,w)
# train_out = np.array(train_out > 0,dtype=int)

# # Compute 0/1 loss on train set
# num_incorrect_train = np.sum(np.abs(train_out - train_bin_label))
# num_correct_train = np.shape(train_out)[0] - num_incorrect_train
# log_loss_train = logLoss(train_bin_label,train_img,w,w0,reg)
# print "0/1 loss (training): %d / %d = %f percent correct." %(num_correct_train,len(train_label),100.0*num_correct_train/len(train_label))
# print "0/1 loss (training): %d / %d = %f percent incorrect." %(num_incorrect_train,len(train_label),100.0*num_incorrect_train/len(train_label))
# print "log loss (training): %f"%log_loss_train


# # Plot log loss
# plt.figure(1)
# plt.plot(range(0,len(lossVecTrain)),lossVecTrain,'b-',range(0,len(lossVecTrain)),lossVecTest,'r-')
# plt.yscale('log')
# plt.xlabel('Iteration')
# plt.ylabel('Log loss')
# plt.legend(['Training set','Test set'])
# plt.title('Log loss for Gradient descent')
# plt.show()

# # Compute loss on test set
# test_out = w0 + np.dot(test_img,w)
# test_out = np.array(test_out > 0,dtype=int)
# num_incorrect_test = np.sum(np.abs(test_out - test_bin_label))
# num_correct_test = np.shape(test_out)[0] - num_incorrect_test
# log_loss_test = logLoss(test_bin_label,test_img,w,w0,reg)
# print "0/1 loss (test): %d / %d = %f percent correct." %(num_correct_test,len(test_label),100.0*num_correct_test/len(test_label))
# print "0/1 loss (test): %d / %d = %f percent incorrect." %(num_incorrect_test,len(test_label),100.0*num_incorrect_test/len(test_label))
# print "log loss (test): %f"%log_loss_test




#-------------------------------------------------------------------------------------------------------------
#									Part 2.2 - Softmax classification with GD
#-------------------------------------------------------------------------------------------------------------

# 
# Create new versions of the functions for Softmax classification
# 

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
		# output[ii*k:((ii+1)*k)] = -np.dot(X.T,(Y==ii+1).astype(int) - P[ii,:])
		output[ii,:] = -np.dot(X.T,(Y==ii+1).astype(int) - P[ii,:])
	return (output) / N + reg * W
	# return (output) / N + reg * W.reshape(len(output))

	# Q = np.empty((nClasses-1,N))
	# for ii in range(1,nClasses):
	# 	Q[ii-1,:] = (Y == ii).astype(int)
	# Q = Q - P
	# output = np.sum(np.kron(Q,X.T),1)

	# for ii in range(0,N):
	# # 	q = np.zeros(nClasses-1)
	# # 	if Y[ii] > 0:
	# # 		q[Y[ii]-1] = 1
	# # 	Q = q - Q
	# # 	output -= np.kron(q - P[:,ii],X[ii,:])

	# 	# Q = getProbs(X[ii,:],W)[1:]
	# 	Q = P[:,ii]
	# 	q = np.zeros(nClasses-1)
	# 	if Y[ii] > 0:
	# 		q[Y[ii]-1] = 1
	# 	Q = q - Q
	# 	output -=  np.kron(Q,X[ii,:])


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
		# t3 = time.time()
		newGrad = Gradient(Ytrain,Xtrain,w,reg)
		# t4 = time.time()

		# print "Time elapsed computing gradient (iteration %d): %f"%(it,t4-t3)


		# # Precompute a possibly expensive quantity
		gradNorm = np.linalg.norm(newGrad,'fro')

		logLossTrain[it] = logLoss(Ytrain,Xtrain,w,reg)
		logLossTest[it] = logLoss(Ytest,Xtest,w,reg)
		z1LossTrain[it] = z1Loss(Ytrain,Xtrain,w)
		z1LossTest[it] = z1Loss(Ytest,Xtest,w)

		# Take a step in the negative gradient direction
		# w = w - step * newGrad.reshape((nClasses-1,Xtrain.shape[1]))
		w = w - step * newGrad

		# t5 = time.time()

		# print "Time elapsed evaluating log loss: %f"%(t5-t4)
		print "Log loss at iteration %d: %f"%(it,logLossTrain[it])
		print "0/1 loss at iteration %d: %f"%(it,z1LossTrain[it])
		print "Order of gradient: %f\n" %np.log10(gradNorm)

		if gradNorm < TOL:
			break

	if (it == MAX_ITS-1):
		print "Warning: Maximum number of iterations reached."

	return (w,logLossTrain[:(it+1)],z1LossTrain[:(it+1)],logLossTest[:(it+1)],z1LossTest[:(it+1)])


# # Set some parameters
# nClasses = 10		# Number of classes
# reg = 1.0			# Regularization parameter

# # Run the method
# (w,logLossTrain,z1LossTrain,logLossTest,z1LossTest) = GD(train_label,test_label,train_img,test_img,reg,nClasses)

# # Plot log loss on test and train sets
# plt.figure(1)
# plt.plot(range(0,len(logLossTrain)),logLossTrain,'b-',range(0,len(logLossTrain)),logLossTest,'r-')
# plt.xlabel('Iteration')
# plt.ylabel('Log loss')
# plt.legend(['Training','Test'])
# # plt.title('Log loss for multi-class logistic regression (batch gradient descent)')
# plt.show()

# # Plot 0/1 loss on the test and train sets
# plt.figure(2)
# plt.plot(range(0,len(z1LossTrain)),z1LossTrain,'b-',range(0,len(logLossTrain)),z1LossTest,'r-')
# plt.xlabel('Iteration')
# plt.ylabel('0/1 loss')
# plt.legend(['Training','Test'])
# # plt.title('0/1 loss for multi-class logistic regression (batch gradient descent)')
# plt.show()

# # Print out final 0/1 and log loss
# print "log loss (training): %f" % (logLossTrain[-1])
# print "0/1 loss (training): %f" % (z1LossTrain[-1])
# print "log loss (test): %f" % (logLossTest[-1])
# print "0/1 loss (test): %f" % (z1LossTest[-1])


# #-------------------------------------------------------------------------------------------------------------
# #									Part 2.3 - Softmax classification with SGD
# #-------------------------------------------------------------------------------------------------------------

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
		# probs[j] = ex[j-1] * denom
	return probs

# Computes an approximate gradient of the log likelihood function using batchSize sample points
def stochGradient(Y,X,W,reg,batchSize):
	N = X.shape[0]
	k = X.shape[1]
	nClasses = W.shape[0]+1
	# output = np.zeros(np.prod(W.shape))
	output = np.zeros(W.shape)

	sampleIndices = np.random.choice(np.arange(0,N),batchSize)
	P = getAProbs(X[sampleIndices,:],W)[1:,:]
	# Y = Y[sampleIndices]

	for ii in range(nClasses-1):
		# output[ii*k:((ii+1)*k)] = -np.dot(X.T,(Y==ii+1).astype(int) - P[ii,:])
		# output[ii,:] = -np.dot(X[sampleIndices,:].T,(Y==ii+1).astype(int) - P[ii,:])
		output[ii,:] = -X[sampleIndices,:].T.dot((Y[sampleIndices]==ii+1).astype(int) - P[ii,:])

	# for ii in sampleIndices:
	# 	Q = getProbs(X[ii,:],W)[1:]
	# 	q = np.zeros(nClasses-1)
	# 	if Y[ii] > 0:
	# 		q[Y[ii]-1] = 1
	# 	Q = q - Q
	# 	output -=  np.kron(Q,X[ii,:])

	return (output / batchSize) + reg * W

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
		

		# print "Time elapsed computing gradient (iteration %d): %f"%(it,t4-t3)


		# # Precompute a possibly expensive quantity
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

			# if np.mod(it,5*outputCond)==0:
			# 	response = raw_input("Reduce step size (y/n/inc)?")
			# 	if response == "y":
			# 		step = step / 4.
			# 	if response == "inc":
			# 		step = step * 2.

			# Average with old iterates
			# w = (w + wOld) / 2.
			# wOld = np.copy(w)

		# Take a step in the negative gradient direction
		step = num / np.sqrt(it+1)
		# w = w - step * newGrad.reshape((nClasses-1,Xtrain.shape[1]))
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

# # # -------------------------------------------------------------------------------------------------------------------
# # Batch size 1: step = 1.e-4 / sqrt(it+1), reg=1.0, about 30 passes over data (450,000 iterations, 294 sec)
# # Can get comparable performance to batch with batchSize=100 and step = (1.e-3 / 4) / sqrt(it+1) in about 20 passes over data (3000 iterations, 198 sec)
# # reg = 1.0

# # Set some parameters
# nClasses = 10		# Number of classes
# reg = 1.0			# Regularization parameter
# batchSize = 1		# Number of points to sample to approximate gradient

# # Add column of all 1's to simulate offset w0
# # train_img = np.concatenate((np.ones((train_img.shape[0],1)),train_img),axis=1)
# # test_img = np.concatenate((np.ones((test_img.shape[0],1)),test_img),axis=1)

# # Run the method
# (w,logLossTrain,z1LossTrain,logLossTest,z1LossTest) = SGD(train_label,test_label,train_img,test_img,reg,nClasses,batchSize)

# # Plot log loss on test and train sets
# n = len(logLossTrain)-1
# outputCond = 15000 / batchSize
# its = range(0,n*outputCond+1,outputCond)
# plt.figure(1)
# plt.plot(its,logLossTrain,'b-o',its,logLossTest,'r-x')
# plt.xlabel('Iteration')
# plt.ylabel('Log loss')
# plt.legend(['Training','Test'])
# # plt.title('Log loss for multi-class logistic regression (stochastic gradient descent)')
# plt.show()

# # Plot 0/1 loss on the test and train sets
# plt.figure(2)
# plt.plot(its,z1LossTrain,'b-o',its,z1LossTest,'r-x')
# plt.xlabel('Iteration')
# plt.ylabel('0/1 loss')
# plt.legend(['Training','Test'])
# # plt.title('0/1 loss for multi-class logistic regression (stochastic gradient descent)')
# plt.show()

# # Print out final 0/1 and log loss
# print "log loss (training): %f" % (logLossTrain[-1])
# print "0/1 loss (training): %f" % (z1LossTrain[-1])
# print "log loss (test): %f" % (logLossTest[-1])
# print "0/1 loss (test): %f" % (z1LossTest[-1])





# #-------------------------------------------------------------------------------------------------------------
# #						Part 2.4 - Softmax classification with SGD - Using more features
# #-------------------------------------------------------------------------------------------------------------

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
# plt.title('Log loss for multi-class logistic regression (stochastic gradient descent)')
plt.show()

# Plot 0/1 loss on the test and train sets
plt.figure(2)
plt.plot(its,z1LossTrain,'b-o',its,z1LossTest,'r-x')
plt.xlabel('Iteration')
plt.ylabel('0/1 loss')
plt.legend(['Training','Test'])
# plt.title('0/1 loss for multi-class logistic regression (stochastic gradient descent)')
plt.show()

# Print out final 0/1 and log loss
print "log loss (training): %f" % (logLossTrain[-1])
print "0/1 loss (training): %f" % (z1LossTrain[-1])
print "log loss (test): %f" % (logLossTest[-1])
print "0/1 loss (test): %f" % (z1LossTest[-1])
