# CSE 546 Homework 2 -- Problem 1: Multi-class Classification using least squares
# Brian de Silva
# 1422824

import numpy as np
from scipy.linalg import lu_factor
from scipy.linalg import lu_solve
from mnist import MNIST
import os

# Import the data
mndata = MNIST('../hw1/mnist')
train_img, train_label = mndata.load_training()
test_img, test_label = mndata.load_testing()

train_img = np.array(train_img)
test_img = np.array(test_img)


# Parameters
tau = 10.0
regularization = np.var(train_label) / (tau * tau)

# Array with columns given by the binary labels
Y = np.zeros((len(train_label),10))
for i in range(0,len(train_label)):
	Y[i,train_label[i]] = 1

YTest = np.zeros((len(test_label),10))
for i in range(0,len(test_label)):
	YTest[i,test_label[i]] = 1

# Convert to floats to prevent rounding issues
train_label = np.array(train_label,dtype=float)
test_label = np.array(test_label,dtype=float)

# Specify which problem we would like to solve (1.1 or 1.2) to avoid writing too complicated a script
problem = 1.2

#-------------------------------------------------------------------------------------------------------------
#									Part 1.1 - One vs all classification
#-------------------------------------------------------------------------------------------------------------
if problem == 1.1:
	print "Solving problem 1.1..."

	# X^T X is expensive to compute, so only do it once
	if os.path.isfile('./XTX.npy'):
		XTX = np.load('XTX.npy')
	else:
		XTX = np.dot(train_img.T,train_img);
		np.save('XTX',XTX)

#-------------------------------------------------------------------------------------------------------------
#							Part 1.2 - Neural nets with random first layer
#-------------------------------------------------------------------------------------------------------------
else:
	print "Solving problem 1.2..."
	k = 10000										# Dimension of new feature vectors

	# Save features for later use (and avoid having to generate them twice)
	if os.path.isfile('./features.npy'):
		V = np.load('features.npy')
	else:
		V = np.random.randn(np.shape(train_img)[1],k)	# Random numbers to generate new features
		np.save('features',V)

	# Generate new features on training data
	train_img = np.dot(train_img,V)
	train_img[train_img < 0] = 0

	# X^T X is expensive to compute, so only do it once
	if os.path.isfile('./XTX_feat.npy'):
		XTX = np.load('XTX_feat.npy')
	else:
		XTX = np.dot(train_img.T,train_img);
		np.save('XTX_feat',XTX)

	# Generate new features on test data
	test_img = np.dot(test_img,V)
	test_img[test_img < 0] = 0


# Solve the problems jointly
I = np.identity(XTX.shape[0])
# (LU,piv) = lu_factor(regularization * I + XTX)
B = np.dot(train_img.T, Y)

# Get all the weights at once
# W = lu_solve((LU,piv),B)
W = np.linalg.solve(regularization * I + XTX,B)

# Get output from all 10 classifiers
train_out = np.dot(train_img,W)

# Decide on classifications for each elt. in training set
train_classes = np.argmax(train_out,1)

# Get associated dot products for computing square loss
train_dots = np.zeros(len(train_classes))
for k in range(0,len(train_dots)):
	train_dots[k] = train_out[k,train_classes[k]]

# Compute 0/1 loss and square loss for training set
train_incorrect = np.count_nonzero(np.around(train_classes - train_label))
train_correct = len(train_label) - train_incorrect

train_sq_loss = np.mean(np.sum((Y - train_out)**2,1))
# train_sq_loss = np.mean(np.square(train_dots - train_label))

print "0/1 loss (training): %d / %d = %f percent correct." %(train_correct,len(train_label),100.0*train_correct/len(train_label))
print "0/1 loss (training): %d / %d = %f percent incorrect." %(train_incorrect,len(train_label),100.0*train_incorrect/len(train_label))
print "Square loss (training): %f" %train_sq_loss

# Classify and get error on testing set

# Get output from all the classifiers
test_out = np.dot(test_img,W)

# Decide on classifications for each elt. in training set
test_classes = np.argmax(test_out,1)

# Get associated dot products for computing square loss
test_dots = np.zeros(len(test_classes))
for k in range(0,len(test_dots)):
	test_dots[k] = test_out[k,test_classes[k]]

# Compute 0/1 loss and square loss for training set
test_incorrect = np.count_nonzero(np.around(test_classes - test_label))
test_correct = len(test_label) - test_incorrect
test_sq_loss = np.mean(np.sum((YTest - test_out)**2,1))
# test_sq_loss = np.mean(np.square(test_dots - test_classes))

print "0/1 loss (testing): %d / %d = %f percent correct." %(test_correct,len(test_label),100.0*test_correct/len(test_label))
print "0/1 loss (testing): %d / %d = %f percent incorrect." %(test_incorrect,len(test_label),100.0*test_incorrect/len(test_label))
print "Square loss (testing): %f" %test_sq_loss