#!/usr/bin/python3

import sys

'''

	This program predicts data on the basis of an input set and determines the best learning rate and 
	other features automatically. You have to give it the number of features that will be implemented.

'''

class gradientDescent:
	def __init__(self, learning_rate = 0.001, features = 1, data_set):
		self.learning_rate = learning_rate
		self.n = features
		self.m = data_set
		# this is the x0 feature which is equal to 1 to model the thetas
		self.x0 = 1
		self.theta = []
		
	def __repr__(self):
		return "Learning Rate : %f \n N : %d \n M : %d \n  Parameters : " \
		% (self.learning_rate, self.n, self.m) + str(self.theta)
	
	def _h(*args):
		sum_var = 0
		for _ in range(self.m):
			sum_var += self.theta[_] * args[_]
		return sum_var
	

	def query(*args):
		return _h(args)


def main():
	learning_rate, features, data_set = tuple(list(map(float, input().strip(' ').split(' '))))
	grad_obj = gradientDescent(learning_rate, features, data_set)



if __name__ == '__main__':
	main()