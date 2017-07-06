#!/home/pratik/anaconda3/bin/python3.6

'''

	This implementation makes use of the np package to handle matrices as well
	as the sklearn SVM package for it's implementation.
	This script will automatically feature scale AS WELL AS choose the best 
	algorithm for the job

'''

# default imports
import config
import helper
import numpy as np
from numpy import linalg
from sklearn import svm
from sys import argv as rd


# custom imports required for the various functions


# returns list of records, 
# as : ( matrix([x1, x2, x3 ... xn], [x1, x2, x3 ... xn] ... [x1, x2, x3 .... xnm]),
# and matrix([y1, y2 ... yK], [y1, y2, ... yk] ... [y1, y2, .... ykm])
class Dataset:

	# intializing the params
	def __init__(self, filename, K):
		self.filename = filename
		self.K = K
		
		# getting the data into the desired form
		raw_data = helper.sanitize(open(filename).read().split(config.r_sep))
		mapped_set = [ [float(_) for _ in x.strip(config.f_sep).split(config.f_sep)] for x in raw_data]
		self.X, self.y = ( np.matrix([np.array(_[:-K]) for _ in mapped_set]), \
			np.matrix([ _[-K:] for _ in mapped_set]) )

		self.X_trans = self.X.T
		self.m, self.n = self.X.shape
		
		# check if multi-class classification
		import resource, sys
		resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
		sys.setrecursionlimit(config.recur_limit)
		uniq = list(set(helper.flatten(np.ndarray.tolist(self.y))))
		self.is_binary = len(uniq) == 2
		self.multi_class = len(uniq) > 2 and len(uniq) <= config.max_K \
			and (len(uniq) / self.m ) <= config.relative_unique

	# prints the dataset
	def printData(self):
		display_all_X = self.X[0].size <= config.max_display_X
		run_count = 1
		for x, y in zip(self.X, self.y):
			if display_all_X:
				print(x, end = "")
			else:
				print(x[0, 0:config.min_display_X], "...", end = "")
			print(" : ", y)
			if run_count >= config.max_example_print:
				print(".",".",".", sep = "\n")
				print("Large data (displaying first %d)" % config.max_example_print)
				break
			run_count += 1

	# this function adds x0, ie 1 to every sample
	def addOnes(self):
		self.X = np.c_[np.ones((self.m, 1)), self.X]
		self.X_trans = self.X.T

	# representing the dataset
	def __repr__(self):
		self.printData()
		return "+--- Dataset Details ---- + \n* Filename : %s \n* (n, m) : (%d, %d) \n* K : %d \
		\n* Binary Data : %s  \n* Multi-Class : %s \n+------------------------ +" \
		% (self.filename, self.n, self.m, self.K, self.is_binary, self.multi_class)


class Algorithm:

	types = ["Linear Regression", "SVM + Linear Kernel OR Logistic Regression", \
		"SVM + Gaussian Kernel", "Add more features", "Neural Network"]
	_cache_size = config.cache_size

	# where d is of type Dataset
	def __init__(self, d):
		# self.types of algorithms that can be run

		# linear regression
		if not d.is_binary and not d.multi_class:
			_type = 0
			# add ones to the dataset
			d.addOnes()
			self.thetas = linalg.pinv(d.X_trans * d.X) * d.X_trans * d.y
		
		# # linear kernel
		# elif d.n <= config.linear_n and d.m <= config.linear_m:
		# 	_type = 1
		# 	print(self.types[_type])
		# 	if d.K == 1:
		# 		self.kernel = 'linear'
		# 		y_temp = [ _[0, 0] for _ in d.y]
		# 		self.machine = svm.SVC(kernel = self.kernel, cache_size = self._cache_size)
		# 	elif d.is_binary:
		# 		y_temp = [np.ndarray.tolist(x)[0].index(1) + 1 for x in d.y]
		# 	else:
		# 		y_temp = [max(np.ndarray.tolist(x)[0]) for x in d.y]
		# 	self.machine.fit(d.X, y_temp)

		# Gaussian Kernel
		elif d.n <= config.gauss_n and d.m <= config.gauss_m:
			_type = 2
			print(self.types[_type])
			# set gaussian kernel
			self.kernel = 'rbf'
			self.C = config.C
			self.gamma = config.gamma

			self.machine = svm.SVC(kernel = self.kernel, C = self.C, \
				gamma = self.gamma, cache_size = self._cache_size)
			if d.K == 1:
				y_temp = [ _[0, 0] for _ in d.y]
			elif d.is_binary:
				y_temp = [np.ndarray.tolist(x)[0].index(1) + 1 for x in d.y]
			else:
				y_temp = [max(np.ndarray.tolist(x)[0]) for x in d.y]
				
			self.machine.fit(d.X, y_temp)
		
		# Add more features
		elif d.n <= config.add_n and d.m >= config.add_m:
			_type = 3
			print(self.types[_type])
			pass
		
		# Neural Network
		else:
			# from Neural import np_initial
			_type = 4
			print(self.types[_type])
			pass
		
		self.d = d
		self.type = _type

	
	# this function is used to predict the output
	def predict(self):
		
		# this is for taking input from the user
		print("Enter the xi(s)", end = " : ")
		xi_s = list(map(float, list(filter( lambda x: x != '', input().strip(' ').split(' ')))))
		print("I/P (x) -> ", xi_s)
		
		# linear use thetas
		if self.type == 0:
			xi_s.insert(0, 1)
			xi_s = np.matrix(xi_s)
			answer = self.thetas * xi_s
		
		# kernel type use svm
		elif self.type in [1, 2]:
			answer = self.machine.predict([xi_s])
			
		print("O/P (y) -> ", answer)
		return answer


	def __repr__(self):
		print(self.d, "\nUsed Algorithm : ", self.types[self.type])
		try:
			print("Thetas obtained : ", self.thetas)
		except:
			pass
		return ""




# main function to get inputs and tie everything together
def main():
	filename = rd[1]
	try:
		K = int(rd[2]) # this is the size of the output vector
	except:
		K = config.def_y_vec_size

	d = Dataset(filename, K)
	alg = Algorithm(d)
	print(alg)

	cont = True
	while cont:
		alg.predict()
		print("Calculate another (Y/n) : ", end = "")
		ans, cont = str(input()), False
		if ans == "" or ans[0] == "" or ans[0] == "y":
			cont = True
	


# Enables script access
if __name__ == '__main__':
	print("SVM Implementation (v1.2)")
	main()
