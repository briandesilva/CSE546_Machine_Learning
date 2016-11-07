# CSE 546 Homework 1 -- Problem 7: Lasso
# Brian de Silva
# 1422824

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix

# Function that evaluates the objective function for debugging purposes
def evalObjFunc(X,y,w,w0,reg):
	if type(X) is np.ndarray:
		return np.linalg.norm(np.dot(X,w) + w0 - y)**2 + reg * np.linalg.norm(w,1)
	else:
		return scipy.sparse.linalg.norm(X*w + w0 - y)**2 + reg * np.linalg.norm(w,1)


#--------------------------------------------------------------------------
# 			Problem 7.2 - Coordinate descent for LASSO
#--------------------------------------------------------------------------

def coordDescent(X,y,reg,w_init=None,w0=0.,TOL=1.e-5,MAX_ITS=1000):
# Performs coordinate descent algorithm to solve Lasso regression problem
# as specified in CSE 546 HW1 problem 7:
# argmin_w {(|Xw + w0 - y|_2)^2 + reg|w|_1}
# Assumption: X is stored in scipy.sparse.csc_matrix format

	# X is N x d
	N = np.shape(X)[0]
	d = np.shape(X)[1]

	# Check if initial guess was passed
	if w_init is None:
		w = np.zeros(d)
	else:
		w = w_init

	delta = 100.0		# maximum amount by which an entry of w changes in an iteration
	its = 0				# count iterations to prevent infinite loop

	if type(X) is np.ndarray:
		while (delta > TOL) and (its < MAX_ITS):
			delta = 0.;
			yhat = np.dot(X,w) + w0			# Update yhat to avoid numerical drift

			w0_new = np.mean(y-yhat) + w0		# Update w0
			yhat += w0_new - w0
			w0 = w0_new
			# yhat = yhat + np.mean(y-yhat)	# Update yhat with new w0

			# Optimize in each coordinate direction
			for k in range(0,d):
				ak = 2.0 * np.linalg.norm(X[:,k])**2
				ck = 2.0 * np.dot(X[:,k],y-yhat) + w[k] * ak;
				if ck < -reg:
					wk_new = (ck + reg) / ak
				elif ck > reg:
					wk_new = (ck - reg) / ak
				else:
					wk_new = 0.

				# Check how much this coordinate of w changed
				if np.abs(wk_new-w[k]) > delta:
					delta = np.abs(wk_new-w[k])
				yhat = yhat + (wk_new - w[k]) * X[:,k]	# Update yhat
				w[k] = wk_new 							# Actually update w[k]
			its = its+1									# Increase iteration count

	# Assume scipy.sparse.csc_matrix type
	else:
		# Initialize a
		a = np.zeros(d)
		for k in range(0,d):
			a[k] = 2.0 * (X[:,k].transpose().dot(X.getcol(k))).toarray()

		while (delta > TOL) and (its < MAX_ITS):
			delta = 0.;
			yhat = X * w + w0					# Update yhat to avoid numerical drift
			w0_new = np.mean(y-yhat) + w0		# Update w0
			yhat += w0_new - w0					# Update yhat with new w0
			w0 = w0_new

			# Optimize in each coordinate direction
			for k in range(0,d):
				# ak = 2.0 * (X[:,k].transpose().dot(X.getcol(k))).toarray()

				ck = 2.0 * X.getcol(k).transpose().dot(y-yhat) + w[k] * a[k]
				if ck < -reg:
					wk_new = (ck + reg) / a[k]
				elif ck > reg:
					wk_new = (ck - reg) / a[k]
				else:
					wk_new = 0.

				# Check how much this coordinate of w changed
				if np.abs(wk_new-w[k]) > delta:
					delta = np.abs(wk_new-w[k])
				yhat = yhat + np.squeeze((X.getcol(k).multiply(np.asscalar(wk_new - w[k]))).toarray() )		# Update yhat
				w[k] = wk_new 											# Actually update w[k]
			its = its+1													# Increase iteration count


	if its == MAX_ITS:
		print "Warning: Maximum number of iterations reached"

	return (w,w0)

#--------------------------------------------------------------------------
# 				Problem 7.3 - testing LASSO with synthetic data
#--------------------------------------------------------------------------

# Parameters
N = 50
d = 75
k = 5
sigma = 10.0
w0Star = 0

# Initialize wstar (true coefficients, the first k of which are nonzero)
wstar = np.zeros(d)
for i in range(0,k/2):
	wstar[i] = 10
for i in range(k/2,k):
	wstar[i] = -10

# Generate data using normal distribution
X = np.random.randn(N,d)

# Generate noise from a normal distribution
eps = sigma * np.random.randn(N)

# Generate artificial data data
y = np.dot(X,wstar) + w0Star + eps

# True sparsity pattern
true_sparsity = np.array(wstar,dtype=bool)


# Solve LASSO problem iteratively to find a good value of lambda
nTrials = 75
lamb = 2 * np.linalg.norm(np.dot(np.transpose(X),y-np.mean(y)),np.inf) * np.ones(nTrials)
for i in range(1,nTrials):
	lamb[i:] = lamb[i:] / 1.1


precision = np.zeros(nTrials)
recall = np.zeros(nTrials)
errNorm = np.zeros(nTrials)
w = np.zeros(d)

for it in range(0,nTrials):
	(w,w0) = coordDescent(X,y,lamb[it],w)
	sparsity_pattern = np.array(w,dtype=bool)
	nMissed = np.sum(true_sparsity+sparsity_pattern) - np.sum(sparsity_pattern)
	nCorrect = np.sum(true_sparsity) - nMissed
	try:
		precision[it] = 1.0 * nCorrect / np.sum(sparsity_pattern)
	except:
		precision[it] = 2.0
	recall[it] = 1.0 * nCorrect / k
	errNorm[it] = np.linalg.norm(np.dot(X,w)+w0-y) / np.linalg.norm(y)

plt.plot(lamb, precision, 'b-o', lamb, recall, 'r-x',lamb,errNorm,'g-s')
plt.xscale('log')
plt.xlabel('$\lambda$')
plt.legend(['Precision','Recall','Relative error'])
plt.axis([lamb[0], lamb[-1], -0.1, 1.1])
plt.title('Precision and Recall as $\lambda$ is decreased')
plt.show()

#---------------------------------------------------------------------------------------
# This code is for the case when sigma = 10

# goodLamb = 340
# # goodLamb = 600 		# Increase lambda to improve performance (remove extra nonzero entries)
# (w,w0) = coordDescent(X,y,goodLamb)
# sparsity_pattern = np.array(w,dtype=bool)
# nMissed = np.sum(true_sparsity+sparsity_pattern) - np.sum(sparsity_pattern)
# nCorrect = np.sum(true_sparsity) - nMissed
# print 'Precision: {}'.format(1.0 * nCorrect / np.sum(sparsity_pattern))
# print 'Recall: {}'.format(1.0 * nCorrect / k)

# # Check solution:
# v = 2 * np.dot(np.transpose(X),np.dot(X,w) + w0 - y)
#---------------------------------------------------------------------------------------



#--------------------------------------------------------------------------
# 				Problem 7.4 - Become a data scientist at Yelp
#--------------------------------------------------------------------------

import scipy.io as io
import scipy.sparse as sparse

# Load a text file of integers:
y = np.loadtxt("hw1-data/upvote_labels.txt", dtype=np.int)

# Load a text file of strings:
featureNames = open("hw1-data/upvote_features.txt").read().splitlines()

# Load a csv of floats:
A = np.genfromtxt("hw1-data/upvote_data.csv", delimiter=",")


# Part 1 - Predicting useful votes

# Normalize data (give all columns 2-norm 1)
A = A / np.linalg.norm(A,2,0)

# Partition the data into training, validation, and test sets
A_train = A[0:4000,:]
A_valid = A[4000:5000,:]
A_test = A[5000:,:]

y_train = y[0:4000]
y_valid = y[4000:5000]
y_test = y[5000:]

# 
# Find a good value of lambda
# 

# Train model on training data
w0 = 0.
w = np.zeros(np.shape(A)[1])
lam = np.linalg.norm(np.dot(np.transpose(A_train),y_train-np.mean(y_train)),np.inf)
TOL = 1.e-3
MAX_ITS = 10000
nTrials = 10

# Some arrays for tracking error, lam values, etc.
rmse_train = np.zeros(nTrials)
rmse_valid = np.zeros(nTrials)
lam_vals = np.zeros(nTrials)
nnz = np.zeros(nTrials)

for k in range(0,nTrials):
	print k

	lam_vals[k] = lam
	(w,w0) = coordDescent(A_train,y_train,lam,w,w0,TOL,MAX_ITS)
	
	# Find RMSE on training data
	rmse_train[k] =  np.sqrt((np.linalg.norm(np.dot(A_train,w) + w0 - y_train)**2) / np.shape(y_train)[0])
	
	# Find RMSE on validation data
	rmse_valid[k] =  np.sqrt((np.linalg.norm(np.dot(A_valid,w) + w0 - y_valid)**2) / np.shape(y_valid)[0])

	# Count number of nonzero weights
	nnz[k] = np.count_nonzero(w)

	lam = lam / 1.5

	if ((k > 0) and ((rmse_valid[k] > rmse_valid[k-1]))):
		break

# Plot results
plt.figure(1)
plt.plot(lam_vals[:k+1], rmse_train[:k+1], 'b-o', lam_vals[:k+1], rmse_valid[:k+1], 'r-x')
plt.xscale('log')
plt.xlabel('$\lambda$')
plt.legend(['Training RMSE','Validation RMSE'])
plt.title('RMSE as $\lambda$ is decreased')

plt.figure(2)
plt.plot(lam_vals[:k+1], nnz[:k+1], 'b-o')
plt.xscale('log')
plt.xlabel('$\lambda$')
plt.title('Number of nonzero weights')
plt.show()

# Check performance on the test set
lam = 3.81944444
w0 = 0
w = np.zeros(np.shape(A)[1])
(w,w0) = coordDescent(A_train,y_train,lam,w,w0,TOL,MAX_ITS)
rmse_test = np.sqrt((np.linalg.norm(np.dot(A_test,w) + w0 - y_test)**2) / np.shape(y_test)[0])

# Print the top 10 most highly weighted features
nzInds = np.nonzero(w)
newInds = np.argsort(np.abs(w[nzInds]))
nzInds = np.squeeze(nzInds)

featInds = nzInds[newInds[-10:]]

print "Top 10 features (in order of ascending importance)"
for ind in featInds:
	print featureNames[ind]



#-----------------------------------------------------------
# Part 2 - Predicting stars
# ----------------------------------------------------------

# Computes the 2-norm of a csc vector
def cscNorm(x):
	y = x.copy()
	y.data **= 2
	return np.sqrt(np.sum(y.data))

# Load a text file of integers:
y = np.loadtxt("hw1-data/star_labels.txt", dtype=np.int)

# Load a matrix market matrix, convert it to csc format:
B = csc_matrix(io.mmread("hw1-data/star_data.mtx"))

# Load a text file of strings:
featureNames = open("hw1-data/star_features.txt").read().splitlines()

# Load

# Normalize the columns of B
d = np.zeros(B.get_shape()[1])
for k in range(0,B.get_shape()[1]):
	d[k]= 1.0 / cscNorm(B.getcol(k))
D = sparse.lil_matrix((B.get_shape()[1],B.get_shape()[1]))
D.setdiag(d)
B = B*D

# Split the data into training, validation, and testing
B_train = B[0:30000,:]
B_valid = B[30000:40000,:]
B_test = B[40000:,:]

y_train = y[0:30000]
y_valid = y[30000:40000]
y_test = y[40000:]

# 
# Find a good value of lambda
# 

# Train model on training data
w0 = 0.
w = np.zeros(B.get_shape()[1])
lam = np.linalg.norm(B_train.transpose() * (y_train-np.mean(y_train)),np.inf)
TOL = 1.e-3
MAX_ITS = 10000
nTrials = 15

# Some arrays for tracking error, lam values, etc.
rmse_train = np.zeros(nTrials)
rmse_valid = np.zeros(nTrials)
lam_vals = np.zeros(nTrials)
nnz = np.zeros(nTrials)

for k in range(0,nTrials):
	print k

	lam_vals[k] = lam
	(w,w0) = coordDescent(B_train,y_train,lam,w,w0,TOL,MAX_ITS)
	
	# Find RMSE on training data
	rmse_train[k] =  np.sqrt((np.linalg.norm(B_train * w + w0 - y_train)**2) / np.shape(y_train)[0])
	
	# Find RMSE on validation data
	rmse_valid[k] =  np.sqrt((np.linalg.norm(B_valid * w + w0 - y_valid)**2) / np.shape(y_valid)[0])

	# Count number of nonzero weights
	nnz[k] = np.count_nonzero(w)

	lam = lam / 1.5

	if ((k > 0) and ((rmse_valid[k] > rmse_valid[k-1]))):
		break

# Plot results
plt.figure(1)
plt.plot(lam_vals[:k+1], rmse_train[:k+1], 'b-o', lam_vals[:k+1], rmse_valid[:k+1], 'r-x')
plt.xscale('log')
plt.xlabel('$\lambda$')
plt.legend(['Training RMSE','Validation RMSE'])
plt.title('RMSE as $\lambda$ is decreased')

plt.figure(2)
plt.plot(lam_vals[:k+1], nnz[:k+1], 'b-o')
plt.xscale('log')
plt.xlabel('$\lambda$')
plt.title('Number of nonzero weights')
plt.show()

# Check performance on the test set
lam = 2.40834045
(w,w0) = coordDescent(B_train,y_train,lam,w,w0,TOL,MAX_ITS)
rmse_test = np.sqrt((np.linalg.norm(B_test * w + w0 - y_test)**2) / np.shape(y_test)[0])

# Print the top 10 most highly weighted features
nzInds = np.nonzero(w)
newInds = np.argsort(np.abs(w[nzInds]))
nzInds = np.squeeze(nzInds)

featInds = nzInds[newInds[-10:]]

print "Top 10 features (in order of ascending importance)"
for ind in featInds:
	print featureNames[ind]