#!/usr/bin/python3
	
'''
	this script can generate datasets for both classification as
	well as liner regression models
	./dataset_gen.py <output filename> < n >  < m > < y (for classification) >

'''

import datetime, sys, os
from random import randint

file_name = sys.argv[1]
n = int(sys.argv[2])
m = int(sys.argv[3])
is_class = "n"

try:
	is_class = sys.argv[4]
	from random import choice, random
	from math import e, ceil
	from statistics import median
except:
	is_class = "n"


if is_class in ["n", "no", "nil", ""]:
	is_class = False

# calculates the regression params for the given run
if is_class:
	theta = [choice([-10, 10]) * random() for i in range(n + 1)]
	print(str(theta))
else:
	theta = [randint(- 10 ** 6, 10 ** 6) for i in range(n + 1)]

# sigmoid function
def sigmoid(h_value):
	# print(1 / (1 + (e ** -h_value)))
	return 1 / (1 + (e ** -h_value))

# to calculate yi on the basis of the thetas and xi
def h(xi_s):
	h_value = sum([i * j for i, j in zip(xi_s, theta)])
	# print(h_value)
	# if is_class:
	# 	return round(sigmoid(h_value / 1000))
	# else:
	return h_value
		

# ties everything together
def main():
	dataset = [ ]
	yi_s = []
	# creating the dataset
	for i in range(m):
		xi_s = [randint(- 10 ** 6, 10 ** 6) for _ in range(n) ]
		if is_class:
			xi_s = [choice([i for i in range(10, 101)]) for _ in range(n)]
		xi_s.insert(0, 1)
		yi_s.append(h(xi_s))
		xi_s.append(yi_s[-1])
		dataset.append(xi_s)
	if is_class:
		mid = median(yi_s)
		dataset = [ (x[:-1] + [0]) if x[-1] < mid else (x[:-1] + [1]) for x in dataset]

	# writing to the file
	if os.path.exists(file_name):
		os.remove(file_name)
	for _  in dataset:
		fout = open(file_name, "a+")
		for i in _:
			fout.write(str(i) + " ")
		fout.write("\n")
		fout.close()
	

# enables script access
if __name__ == '__main__':
	main()