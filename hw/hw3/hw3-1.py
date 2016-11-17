# CSE 546 Homework 3
# Problem 1: PCA and Reconstruction 

# Brian de Silva
# 1422824

import numpy as np
from mnist import MNIST
import os
import time
import matplotlib.pyplot as plt


# Import the data
mndata = MNIST('../hw1/mnist')
train_img, train_label = mndata.load_training()
# test_img, test_label = mndata.load_testing()

# Convert to numpy arrays
train_img = np.array(train_img)
train_label = np.array(train_label,dtype=int)


# #-------------------------------------------------------------------------------------------------------------
# #											Part 1.2 - PCA
# #-------------------------------------------------------------------------------------------------------------

# We don't actually need to compute Sigma, can just go straight to SVD of X
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

# Print a few eigenvalues of Sigma
n = train_img.shape[0]
eigs = S**2 / n
print "1st eigenvalue of Sigma: %f" %(eigs[0])
print "2nd eigenvalue of Sigma: %f" %(eigs[1])
print "10th eigenvalue of Sigma: %f" %(eigs[9])
print "30th eigenvalue of Sigma: %f" %(eigs[29])
print "50th eigenvalue of Sigma: %f" %(eigs[49])

# Print sum of eigenvalues
eigSum = np.sum(eigs)
print "Sum of eigenvalues of Sigma: %f" %(eigSum)

# Compute and plot fractional reconstruction error
fre = np.empty(50)
for k in range(0,50):
	fre[k] = 1. - np.sum(eigs[:(k+1)]) / eigSum

plt.figure(1)
plt.plot(range(0,50),fre,'b-o')
plt.xlabel('Number of PCA directions used')
plt.ylabel('Fractional reconstruction error')
plt.title('Fractional reconstrution error')

# #-------------------------------------------------------------------------------------------------------------
# #								Part 1.3 - Visualization of the Eigen-directions
# #-------------------------------------------------------------------------------------------------------------

# Plot the first 10 eigenvectors as images
imSize = np.sqrt(train_img.shape[1]).astype(int)
plt.figure(2)
for k in range(0,10):
	plt.subplot(2,5,k+1)
	plt.title("%d"%(k+1))
	imgplot = plt.imshow(V[k,:].reshape((imSize,imSize)))
	imgplot.set_cmap('Greys')
	plt.axis('off')


# #-------------------------------------------------------------------------------------------------------------
# #								Part 1.4 - Visualization and Reconstruction
# #-------------------------------------------------------------------------------------------------------------

# The first five images are of different digits (5, 0, 4, 1, 9), so use those
plt.figure(3)
for k in range(0,5):
	plt.subplot(1,5,k+1)
	imgplot = plt.imshow(train_img[k,:].reshape((imSize,imSize)))
	imgplot.set_cmap('Greys')
	plt.axis('off')


# Reconstruct the images using various numbers of eigenvectors
figCount = 4
for k in [2,5,10,20,50]:
	reconstruction = np.dot(V[:k,:].T.dot(V[:k,:]),train_img[0:5,:].T)
	plt.figure(figCount)
	for j in range(0,5):
		plt.subplot(1,5,j+1)
		imgplot = plt.imshow(reconstruction[:,j].reshape((imSize,imSize)))
		imgplot.set_cmap('Greys')
		plt.axis('off')
	figCount += 1
plt.show()