# CSE 546 Homework 2
# Problem 2.1: Binary logistic regression

# Brian de Silva
# 1422824

import numpy as np
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
#									Part 2.1 - Binary Logistic Regression
#-------------------------------------------------------------------------------------------------------------


# Weight vector
w = np.zeros(np.shape(train_img)[1])
w0 = 0.0

# Regularization parameter
reg = 1.0

# Log loss (function we're trying to minimize)
def logLoss(Y,X,w,w0,reg):
	Xw = np.dot(X,w)
	c1 = np.dot(Y.T,Xw + w0)
	c2 = np.sum(np.log(1 + np.exp(w0 + Xw)))
	c3 = (reg / 2.0) * np.linalg.norm(w)**2
	return (c2 + c3 - c1)/len(Y)

# Computes the full gradient of the log likelihood function
def fullGradient(Y,X,w,w0,reg):
	fullGrad = np.empty(len(w)+1)
	C1 = np.exp(w0 + np.dot(X,w))
	C2 = Y - (C1 / (1.0 + C1))
	fullGrad[0] = np.sum(C2)
	fullGrad[1:] = (np.dot(X.T,C2) - reg * w) 
	return fullGrad / len(Y)

# Use Gradient Descent to miminimize log loss
def GD(Ytrain,X,Ytest,Xtest,reg,w=None,w0=0.,TOL=1.e-2,MAX_ITS=500):
	if w is None:
		w = np.zeros(np.shape(X)[1])

	N = len(X)
	step = 1.e-5
	lossVecTrain = np.empty(MAX_ITS)
	lossVecTest = np.empty(MAX_ITS)

	for it in range(0,MAX_ITS):
		newGrad = fullGradient(Ytrain,X,w,w0,reg)

		lossVecTrain[it] = logLoss(Ytrain,X,w,w0,reg)
		
		# Take a step in gradient direction
		w = w + step * newGrad[1:]
		w0 = w0 + step * newGrad[0]

		lossVecTest[it] = logLoss(Ytest,Xtest,w,w0,reg)

		print "Log loss at iteration %d: %f"%(it,lossVecTrain[it])
		print "Order of gradient: %f\n" %np.log10(np.linalg.norm(newGrad,np.inf))

		if np.linalg.norm(newGrad,np.inf) < TOL:
			break

	if (it == MAX_ITS-1):
		print "Warning: Maximum number of iterations reached."

	return (w,w0,lossVecTrain[:it],lossVecTest[:it])

(w,w0,lossVecTrain,lossVecTest) = GD(train_bin_label,train_img,test_bin_label,test_img,reg)

# Classify the points
train_out = w0 + np.dot(train_img,w)
train_out = np.array(train_out > 0,dtype=int)

# Compute 0/1 loss on train set
num_incorrect_train = np.sum(np.abs(train_out - train_bin_label))
num_correct_train = np.shape(train_out)[0] - num_incorrect_train
log_loss_train = logLoss(train_bin_label,train_img,w,w0,reg)
print "0/1 loss (training): %d / %d = %f percent correct." %(num_correct_train,len(train_label),100.0*num_correct_train/len(train_label))
print "0/1 loss (training): %d / %d = %f percent incorrect." %(num_incorrect_train,len(train_label),100.0*num_incorrect_train/len(train_label))
print "log loss (training): %f"%log_loss_train


# Plot log loss
plt.figure(1)
plt.plot(range(0,len(lossVecTrain)),lossVecTrain,'b-',range(0,len(lossVecTrain)),lossVecTest,'r-')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Log loss')
plt.legend(['Training set','Test set'])
plt.title('Log loss for Gradient descent')
plt.show()

# Compute loss on test set
test_out = w0 + np.dot(test_img,w)
test_out = np.array(test_out > 0,dtype=int)
num_incorrect_test = np.sum(np.abs(test_out - test_bin_label))
num_correct_test = np.shape(test_out)[0] - num_incorrect_test
log_loss_test = logLoss(test_bin_label,test_img,w,w0,reg)
print "0/1 loss (test): %d / %d = %f percent correct." %(num_correct_test,len(test_label),100.0*num_correct_test/len(test_label))
print "0/1 loss (test): %d / %d = %f percent incorrect." %(num_incorrect_test,len(test_label),100.0*num_incorrect_test/len(test_label))
print "log loss (test): %f"%log_loss_test
