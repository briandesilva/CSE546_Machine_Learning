# CSE 546 Homework 3
# Problem 5: K-means clustering

# Brian de Silva
# 1422824


import numpy as np
from mnist import MNIST
import os
import time
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats 		# For stats.mode

# Time the code
t1 = time.time()

# 
# Import the data
# 

mndata = MNIST('../hw1/mnist')
train_img, train_label = mndata.load_training()
test_img, test_label = mndata.load_testing()

# Convert to numpy arrays
train_img = np.array(train_img)
train_label = np.array(train_label,dtype=int)
test_img = np.array(test_img)
test_label = np.array(test_label,dtype=int)


# ---------------------------------------------------------------------------------------------------
# 										Function definitions
# ---------------------------------------------------------------------------------------------------


# ---K-means---
# 
# Naive K-means implementation
# 	Input:
# 		-numClasses is the number of clusters
# 		-X is the data matrix (rows are data points)
# 		-MAX_ITS is the maximum number of iterations taken before k-means terminates
# 	Output:
# 		-Y is the set of labels for points in X (labels take values 0,1,...,numClasses-1)
# 		-Mu is the set of means of the classes (ceners[i] corresponds to class i)
# 		-reconErr2 is the square reconstruction error at each iteration
def kmeans(numClasses,X,MAX_ITS=300):
	N = X.shape[0]						# Number of data points
	Y = np.zeros(N).astype(int)			# Labels
	Yprev = np.zeros(N).astype(int)		# Labels from previous iteration
	reconErr2 = np.zeros(MAX_ITS)		# Square reconstruction error

	# TODO: Make sure none of these are too close together
	# Initialize centers as numClasses random points in data
	Mu = X[np.random.choice(np.arange(N),numClasses,replace=False),:]

	# Store square 2-norms of data points to speed up computation
	X2 = np.sum(X**2,1)

	# Iteratively update centers until convergence
	for it in range(MAX_ITS):

		# Compute quantities we can reuse
		Mu2 = np.sum(Mu**2,1)

		# Assign labels and get error (starts at iteration 0)
		for k in range(N):
			dists = Mu2 + X2[k] - 2. * Mu.dot(X[k,:])
			Y[k] = np.argmin(dists)
			reconErr2[it] += dists[Y[k]]

		# Compute new centers
		for k in range(numClasses):
			Mu[k,:] = np.mean(X[Y==k,:],0)

		# Check if we should exit
		if np.array_equal(Y,Yprev):
			break
		else:
			Yprev = np.copy(Y)

	print "%d iterations of k-means to achieve convergence." %(it+1)

	return (Y,Mu,reconErr2[:(it+1)])


# ---0/1 Loss---
# 
# 	Computes the 0/1 loss as a percentage
# 		-YTrue is set of true labels
# 		-Y is set of empirical labels
def z1Loss(YTrue,Y):
	return (1. * np.count_nonzero(YTrue - Y)) / len(YTrue)


# ---------------------------------------------------------------------------------------------------
#										5.1 Run the algorithm
# ---------------------------------------------------------------------------------------------------

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

# Rescale the data so no overflow errors are encountered
scaleFactor = np.sqrt(np.max(np.sum(trainProj**2,1)))
trainProj /= scaleFactor
testProj /= scaleFactor

# 
# Run k-means with 16 centers
# 

numClusts = 16
t1 = time.time()
(labels,means,sqErr) = kmeans(numClusts,trainProj)
print "Time elapsed to run k-means: %f" %(time.time()-t1)

# Rescale the centers
means *= scaleFactor

# Plot the square reconstruction error
matplotlib.rcParams.update({'font.size' : 22})
plt.figure(1)
plt.plot(range(1,len(sqErr)),sqErr[1:],'b',linewidth=2.0)
plt.xlabel('Iteration')
plt.ylabel('Square reconstruction error (16 centers)')

# Plot the number of assignments for each center
(hist,dummy) = np.histogram(labels,bins=range(numClusts+1))
inds = np.argsort(-hist)
histSort = -np.sort(-hist)

plt.figure(2)
plt.bar(np.arange(numClusts),histSort,align='center',alpha=0.5)
plt.xticks(np.arange(numClusts),inds)
plt.ylabel('Number of class assignments')
plt.xlabel('Class')

# Project back up to the higher dimensional space
projMeans = means.dot(V[:50,:])

# Visualize the 16 centers
imSize = np.sqrt(projMeans.shape[1]).astype(int)
plt.figure(3)
for k in range(16):
	plt.subplot(4,4,k+1)
	imgplot = plt.imshow(projMeans[k,:].reshape(imSize,imSize))
	imgplot.set_cmap('Greys')
	plt.axis('off')
plt.show()

# 
# Try redo-ing the above with 250 centers
# 

numClusts = 250

t1 = time.time()
(labels,means,sqErr) = kmeans(numClusts,trainProj)
print "Time elapsed to run k-means: %f" %(time.time()-t1)

# Rescale the centers
means *= scaleFactor

# Plot the square reconstruction error
plt.figure(4)
plt.plot(range(1,len(sqErr)),sqErr[1:],'b',linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Square reconstruction error (250 centers)')

# Plot the number of assignments for each center
(hist,dummy) = np.histogram(labels,bins=range(numClusts+1))
inds = np.argsort(-hist)
histSort = -np.sort(-hist)

plt.figure(5)
plt.bar(np.arange(numClusts),histSort,align='edge',alpha=0.5)
plt.ylabel('Number of class assignments')

# Project back up to the higher dimensional space
projMeans = means.dot(V[:50,:])

# Visualize 16 random centers

imSize = np.sqrt(projMeans.shape[1]).astype(int)
samples = np.random.choice(np.arange(0,numClusts),16)
sampledMeans = projMeans[samples,:]
sampledHist = hist[samples]
sampledInds = np.argsort(-sampledHist)

plt.figure(6)
for k in range(16):
	plt.subplot(4,4,k+1)
	imgplot = plt.imshow(sampledMeans[sampledInds[k],:].reshape(imSize,imSize))
	imgplot.set_cmap('Greys')
	plt.axis('off')
	# plt.title(samples[k])
plt.show()


# ---------------------------------------------------------------------------------------------------
#										5.2 Classification with K-means
# ---------------------------------------------------------------------------------------------------


# 
# Classify with 16 clusters
# 

numClusts = 250

(labels,means,sqErrTrain) = kmeans(numClusts,trainProj,500)

# Determine labels for each mean
meanLabels = np.empty(numClusts)
meanLabelsTest = np.empty(numClusts)
for k in range(numClusts):
	meanLabels[k] = stats.mode(train_label[labels==k])[0][0]

# Classify each point by checking which mean it is closest to then giving it that mean's label
kmeansClassTrain = np.empty(len(train_label))
for k in range(len(train_label)):
	kmeansClassTrain[k] = meanLabels[np.argmin(np.sum((means-trainProj[k,:])**2,1))]

kmeansClassTest = np.empty(len(test_label))
for k in range(len(test_label)):
	kmeansClassTest[k] = meanLabels[np.argmin(np.sum((means-testProj[k,:])**2,1))]

# Compute 0/1 loss
z1Train = (1.0 * np.count_nonzero(kmeansClassTrain - train_label)) / train_label.shape[0]
z1Test = (1.0 * np.count_nonzero(kmeansClassTest - test_label)) / test_label.shape[0]

print "0/1 loss for k-means with %d means (train): %f" %(numClusts,z1Train)
print "0/1 loss for k-means with %d means (test): %f" %(numClusts,z1Test)