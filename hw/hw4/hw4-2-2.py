# CSE 546 Homework 4
# Problem 2.2: Neural Nets and Backprop (ReLu hidden units)

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

numClasses = 10

# ----------------------------------------------------------------------------------------
# 								Function defintions
# ----------------------------------------------------------------------------------------


# Neural network class using ReLu activation function at hidden layer
class NN:
	# Constructor
	# 
	# Size of network is specified in a tuple:
	# n = (input dim, # hidden nodes, numClasses)
	# 
	# and scaling factor is set in a tuple Scalefact:
	# Scalefact = (scale factor for W_i, factor for W_h)
	def __init__(self,n,scaleFact=None):
		if scaleFact==None:
			scaleFact = (np.sqrt(n[0]),n[0])
		self.W_i = np.random.randn(n[1],n[0]) / scaleFact[0]
		self.W_h = np.random.randn(n[2],n[1]) / scaleFact[1]

	# Generates output from one forward pass through the neural net
	# Data should be contained in COLUMNS of X
	def forwardPass(self,X):
		self.hz = self.W_i.dot(X)				# Hidden layer input
		self.ha = np.maximum(self.hz,0)			# Hidden layer activation
		self.oa = self.W_h.dot(self.ha)			# Output layer activation

	# Computes the gradient of the square loss wrt the weights in the NN
	# Y should be a matrix with each column a length 10 one-hot vector of class labels
	# steps := (stepsize for W_i, stepsize for W_h)
	def backProp(self,X,Y,steps):
		N = X.shape[1]
		dW_h = -(Y - self.oa)     				# note: activation function is identity
		dW_i = self.W_h.T.dot(dW_h) * (self.hz > 0).astype(float)
		self.W_h -= steps[1] * dW_h.dot(self.ha.T) / N
		self.W_i -= steps[0] * dW_i.dot(X.T) / N

	# A function that returns the gradients of the two weight matrices
	def getGrads(self,X,Y):
		self.forwardPass(X)
		N = X.shape[1]
		dW_h = -(Y - self.oa)      				# note: activation function is identity
		dW_i = self.W_h.T.dot(dW_h) * (self.hz > 0).astype(float)
		return (dW_i.dot(X.T) / N, dW_h.dot(self.ha.T) / N)


	# Store the weights in files in case we need to re-use them
	def checkpoint(self,fnamePrefix,fnameSuffix=""):
		np.save(fnamePrefix + "W_i" + fnameSuffix,self.W_i)
		np.save(fnamePrefix + "W_h" + fnameSuffix,self.W_h)


	# Compute the mean square loss
	# Y should be a matrix with each column a length 10 one-hot vector of class labels
	def squareLoss(self,X,Y):
		self.forwardPass(X)
		return np.mean(np.sum((Y - self.oa)**2,0))

	# Compute the 0/1 loss
	# Y should be a vector of class labels
	def z1Loss(self,X,Y):
		self.forwardPass(X)
		return (1.0 * np.count_nonzero(np.argmax(self.oa,0) - Y)) / Y.shape[0]

	# Compute mean square loss and 0/1 loss
	# Y should be a matrix with each column a length 10 one-hot vector of class labels
	def getLoss(self,X,Y):
		self.forwardPass(X)

		# Compute square loss
		sql = np.mean(np.sum((Y - self.oa)**2,0))

		# Compute 0/1 loss
		z1l = 1. * np.count_nonzero(np.argmax(Y,0) != np.argmax(self.oa,0)) / Y.shape[1]

		return (sql,z1l)



# -----------------------------------------------------------------------------------------

# Compute SVD
if os.path.isfile("V.npy"):
	V = np.load("V.npy")
else:
	(U,S,V) = np.linalg.svd(train_img,False)
	np.save("V",V)

# **Note** This V is actually V.H in the traditional SVD: X = U*S*V.H

# Project onto the first 50 singular vectors
trainProj = train_img.dot(V[:50,:].T)
testProj = test_img.dot(V[:50,:].T)
# Note: to get true projection, do trainProj.dot(V[:50,:])

# Create binary labels
trainBinLabels = np.zeros((10,len(train_label)))
testBinLabels = np.zeros((10,len(test_label)))
labVals = range(10)
for k in range(train_label.shape[0]):
	trainBinLabels[:,k] = (train_label[k] == labVals)
for k in range(test_label.shape[0]):
	testBinLabels[:,k] = (test_label[k] == labVals)


# Set parameters
# --------------------------------------------------------------------------
nClasses = 10			# Number of classes
reg = 0.				# Regularization parameter
batchSize = 10			# Batch size for SGD
numEpochs = 30			# Number of epochs to go through before terminating
nnSize = (50,500,10)	# Number of nodes in each layer of NN (input, hidden, output)
step = 1.e-3			# Learning rate
std = (np.sqrt(np.mean(trainProj**2)),np.sqrt(50))	# Initial weight standard deviations
ckptStr = "init_test2.1"	# String used in checkpointing filename
ckptFreq = numEpochs
numLargeSteps = 10

print "Time elapsed during setup:\t\t %f" %(time.time() - t1)
# --------------------------------------------------------------------------


# 
# Train the network
# 

# Initialize the neural net
nn = NN(nnSize,std)

nn.forwardPass(trainProj.T)


# Rescale the weights so that E(Yhat) ~ E(Y) / 10
# Note: E(Y) = 1/10
nn.W_h /= (100*np.mean(nn.oa))
nn.W_i /= (100*np.mean(nn.oa))


# Create vectors in which to store the loss
trainLoss = np.zeros((2,2 * numEpochs + 1))		# First row is sq loss, 2nd is 0/1
testLoss = np.zeros((2,2 * numEpochs + 1))		# First row is sq loss, 2nd is 0/1

# SGD
N = train_img.shape[0]

# Loop over epochs
for it in range(numEpochs):
	t2 = time.time()

	# Check if we need to reduce the stepsize
	if (np.mod(it,numLargeSteps)==0 and it > 0):
	# if it == numLargeSteps:
		step /= 4

	# Get new sampling order for this epoch
	sampleIndices = np.random.permutation(N)

	# Compute losses
	trainLoss[:,2*it] = nn.getLoss(trainProj.T,trainBinLabels)
	testLoss[:,2*it] = nn.getLoss(testProj.T,testBinLabels)

	# Print losses
	print "Square loss after %d epochs:\t\t %f"%(it,trainLoss[0,2*it])
	print "0/1 loss after %d epochs:\t\t %f\n"%(it,trainLoss[1,2*it])

	# Check the norms of the gradients
	(gi,gh) = nn.getGrads(trainProj.T,trainBinLabels)
	# print "Orders of gradients W_i, W_h after %d epochs:\t %f, %f\n"%(it,np.log(np.sqrt(np.sum(gi**2))),np.log(np.sqrt(np.sum(gh**2))))

	# Take multiple gradient steps for half an epoch
	for subIt in range((N / batchSize) / 2):
		# Get indices for this iteration
		currIndices = sampleIndices[subIt*batchSize:(subIt+1)*batchSize]

		# Forward pass
		nn.forwardPass(trainProj[currIndices,:].T)

		# Update weights
		nn.backProp(trainProj[currIndices,:].T,trainBinLabels[:,currIndices],(step,step))



	# Compute and print losses after half an epoch
	trainLoss[:,2*it+1] = nn.getLoss(trainProj.T,trainBinLabels)
	testLoss[:,2*it+1] = nn.getLoss(testProj.T,testBinLabels)
	print "Square loss after %d 1/2 epochs:\t %f"%(it,trainLoss[0,2*it+1])
	print "0/1 loss after %d 1/2 epochs:\t\t %f\n"%(it,trainLoss[1,2*it+1])

	# Check the norms of the gradients
	(gi,gh) = nn.getGrads(trainProj.T,trainBinLabels)
	# print "Orders of gradients W_i, W_h after %d 1/2 epochs:\t %f, %f\n"%(it,np.sqrt(np.sum(gi**2)),np.sqrt(np.sum(gh**2)))

	# Finish the epoch
	for subIt in range((N / batchSize) / 2, N / batchSize):
		# Get indices for this iteration
		currIndices = sampleIndices[subIt*batchSize:(subIt+1)*batchSize]

		# Forward pass
		nn.forwardPass(trainProj[currIndices,:].T)

		# Update weights
		nn.backProp(trainProj[currIndices,:].T,trainBinLabels[:,currIndices],(step,step))


	# Output elapsed time
	print "Time elapsed during epoch %d:\t\t %f"%(it, time.time() - t2)
	print "-----------------------------------------------------------------------\n\n"

	# Check if we need to checkpoint
	if (it > 0 and np.mod(it,ckptFreq)==0):
		nn.checkpoint(ckptStr,"Ep" + str(it+1))

	# Get final losses
	if it == numEpochs-1:
		trainLoss[:,2*(it+1)] = nn.getLoss(trainProj.T,trainBinLabels)
		testLoss[:,2*(it+1)] = nn.getLoss(testProj.T,testBinLabels)


# Print final losses
print "Train Square loss after %d epochs:\t %f"%(numEpochs,trainLoss[0,-1])
print "Train 0/1 loss after %d epochs:\t\t %f"%(numEpochs,trainLoss[1,-1])
print "Test Square loss after %d epochs:\t %f"%(numEpochs,testLoss[0,-1])
print "Test 0/1 loss after %d epochs:\t\t %f"%(numEpochs,testLoss[1,-1])

print "Total time elapsed to run %d epochs:\t %f"%(numEpochs,time.time() - t1)


# Plot the loss
matplotlib.rcParams.update({'font.size' : 20})
its = np.arange(0,numEpochs+0.5,0.5)

# Square loss
plt.figure(1)
plt.plot(its,trainLoss[0,:],'b-o',its,testLoss[0,:],'r-x',linewidth=1.5)
plt.xlabel('Epoch')
plt.ylabel('Square loss')
plt.legend(['Training','Test'])

# 0/1 loss
plt.figure(2)
plt.plot(its[1:],trainLoss[1,1:],'b-o',its[1:],testLoss[1,1:],'r-x',linewidth=1.5)
plt.xlabel('Epoch')
plt.ylabel('0/1 loss')
plt.legend(['Training','Test'])
plt.show()

# 
# Visualize 10 random hidden layer weights
# 

# Select 10 random hidden node weights to visualize
nodeIndices = np.random.choice(np.arange(500),10,replace=False)
nodes = nn.W_i[nodeIndices,:]

# Project up to higher dimensional space
projNodes = nodes.dot(V[:50,:])

# Visualize the weights
imSize = np.sqrt(projNodes.shape[1]).astype(int)
plt.figure(3)
for k in range(10):
	plt.subplot(2,5,k+1)
	imgplot = plt.imshow(projNodes[k,:].reshape(imSize,imSize))
	imgplot.set_cmap('Greys')
	plt.axis('off')

plt.show()	