# CSE 546 Homework 1 -- Problem 6: Classification using least squares
# Brian de Silva
# 1422824

import numpy as np
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt
from mnist import MNIST
import os

# Import the data
mndata = MNIST('mnist')
train_img, train_label = mndata.load_training()
test_img, test_label = mndata.load_testing()

# Make the labels binary
for i in range(0,np.shape(train_label)[0]):
	if (train_label[i] !=2):
		train_label[i] = 0
	else:
		train_label[i] = 1

for i in range(0,np.shape(test_label)[0]):
	if (test_label[i] !=2):
		test_label[i] = 0
	else:
		test_label[i] = 1

train_label = np.array(train_label,dtype=float)
test_label = np.array(test_label,dtype=float)

# Solve least-squares problem to get weights w
tau = 10.0
regularization = np.var(train_img) / (tau * tau)

# # X^T X is expensive to compute, so only do it once
# if os.path.isfile('./XTX.npy'):
# 	XTX = np.load('XTX.npy')
# else:
# 	XTX = np.dot(np.transpose(train_img),train_img);
# 	np.save('XTX',XTX)


# I = np.identity(np.shape(XTX)[0])
# w = np.linalg.solve(regularization * I + XTX, np.dot(np.transpose(train_img), train_label))

# See if we can get equivalent solution with QR-factorization
# We can--save all these so that we don't need to generate every time
if os.path.isfile('./Xhat.npy'):
	Xhat = np.load('Xhat.npy')
else:
	Xhat = np.concatenate(((train_img / np.std(train_img)),np.identity(np.shape(train_img)[1]) / tau))
	np.save('Xhat',Xhat)
if os.path.isfile('./Q.npy'):
	Q = np.load('Q.npy')
	R = np.load('R.npy')
else:
	(Q,R) = np.linalg.qr(Xhat)
	np.save('Q',Q)
	np.save('R',R)

# Xhat = np.concatenate(((train_img / np.std(train_img)),np.identity(np.shape(train_img)[1]) / tau))
# yhat = np.concatenate((train_label / np.std(train_img),np.zeros(np.shape(train_img)[1])))
# (Q,R) = np.linalg.qr(Xhat)
w = solve_triangular(R, np.dot(np.transpose(Q),np.concatenate((train_label / np.std(train_img),np.zeros(np.shape(train_img)[1])))))

# Classify the images in training set
threshold = np.linspace(0,0.5,1000)[759]
train_apx = np.dot(train_img,w)
# save_train_apx = np.copy(train_apx)		# For finding best threshold
true_dot = train_apx[np.nonzero(train_label)]
false_dot = train_apx[~(np.array(train_label,dtype=bool))]

#-------------------------------------------------------------------------------------------------------------
#	This code was for finding a good threshold value
#-------------------------------------------------------------------------------------------------------------
# # Plot a histogram of the dot products to help determine a good range of thresholds to try
# plt.figure(1)
# plt.hist(true_dot)
# plt.title("Dot products for the 2's")

# plt.figure(2)
# plt.hist(false_dot)
# plt.title("Dot products for the other images")
# plt.show()

# # Test many values of threshold to find the best one
# sq_loss = np.zeros(1000)
# count = 0
# for threshold in np.linspace(0,0.5,1000):
# 	train_apx = np.copy(save_train_apx)
# 	for i in range(0,np.shape(train_apx)[0]):
# 		if (train_apx[i] > threshold):
# 			train_apx[i] = 1
# 		else:
# 			train_apx[i] = 0

# 	# Compute 0/1 loss
# 	num_incorrect_train = np.sum(np.abs(train_apx - train_label))
# 	num_correct_train = np.shape(train_apx)[0] - num_incorrect_train


# 	# Compute square loss
# 	sq_loss_train = np.mean(np.square(train_label-train_apx))
# 	sq_loss[count] = sq_loss_train
# 	count += 1

# threshold = np.linspace(0,0.5,1000)[np.argmin(sq_loss)]
#-------------------------------------------------------------------------------------------------------------


for i in range(0,np.shape(train_apx)[0]):
	if (train_apx[i] > threshold):
		train_apx[i] = 1
	else:
		train_apx[i] = 0

# Compute 0/1 loss on training set
num_incorrect_train = np.sum(np.abs(train_apx - train_label))
num_correct_train = np.shape(train_apx)[0] - num_incorrect_train
sq_loss_train = np.mean(np.square(train_label-train_apx))


# Get error on the testing set
test_apx = np.dot(test_img,w)

for i in range(0,np.shape(test_apx)[0]):
	if (test_apx[i] > threshold):
		test_apx[i] = 1
	else:
		test_apx[i] = 0

num_incorrect_test = np.sum(np.abs(test_apx - test_label))
num_correct_test = np.shape(test_apx)[0] - num_incorrect_test

# Compute square loss
sq_loss_test = np.mean(np.square(test_label-test_apx))