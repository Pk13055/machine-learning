#!/usr/bin/python3


# Data set Generator
'''
	
	this python script will create datasets randomly on the basis of the number of features and 
	the data inputs that you require. You can further use these values to test out mean normalization
	as well as feature scaling

'''

import sys
import random

# the number of features
n = int(sys.argv[1])
# the number of data inputs
m = int(sys.argv[2])
file_print = sys.argv[3]
try:
	file_name = sys.argv[4]
except:
	file_name = "dataset.txt"
try:
	threshold = int(sys.argv[5])
except:
	threshold = 10 ** 8

data_set = []

# data will be modelled as i => 0 to n where 1 to n are the features


# this creates an [1, .. , ] feature set for each training example
def create_features(n):
	features = [1]
	ranges = []
	# range generation
	for _ in range(n):
		lower = random.randint(1, 10 ** 9)
		upper = random.randint(lower, 10 ** 9)
		if upper - lower > threshold:
			lower = (upper / (2 * lower) ) * lower
		ranges.append(tuple([int(lower), upper]))
	print(ranges)
	# data generation
	for _ in range(n):
		features.append(random.randint(ranges[_][0], ranges[_][1]))
	return features

# main function which ties everything together
def main():
	for _ in range(m):
		yi = random.randint(threshold, 10 ** 9)
		data_set.append([ create_features(n), yi ])
	if file_print in ["y", "yes", "Y", "YES", "Yes"]:
		fin = open(file_name, "w")
		for x in data_set:
			for i in x[0]:
				fin.write(str(i) + " ")
			fin.write(str(x[1]) + "\n")
		fin.close()
	return data_set


# enables script access
if __name__ == '__main__':
	main()