#!/usr/bin/python3

'''

	this script is responsible for normalizing data and keeping track of the various params used
	It accepts the file as a parameter and applies mean subtraction as well as feature scaling 
'''

import sys
from statistics import stdev, mean
# Data should be of the form 
# feature 1 feature 2 feature 3 ... feature n + 1 cost
try:
	file_name = sys.argv[1]
except:
	file_name = "dataset.txt"

# this function removes any oddities in the data set
def sanitize(unset_data):
	return type(unset_data)(filter(lambda x: x != '', unset_data))

# this function extracts the feature-set from the data so that it can be normalized
def process(unset_data):
	m = len(unset_data)
	n = len(sanitize(unset_data[0].split(' '))) - 1
	feature_sets = [ [] for _ in range(n) ]
	yi = []
	for _ in range(m):
		unset_data[_] = unset_data[_].split(' ')
		for i in range(n):
			feature_sets[i].append(unset_data[_][i])
		yi.append(unset_data[_][-1])
	return tuple([feature_sets, yi])

def main():
	unset_data = open(file_name).read().strip('\n').split('\n')
	feature_sets, yi_s = process(sanitize(unset_data))
	normalized_set = []
	for _ in feature_sets[1:]:
		_ = list(map(int, _))
		u_i = int(mean(_))
		dev = int(stdev(_))
		print(u_i, dev)
		_ = list(map(lambda x: (x - u_i) / dev, _))
		normalized_set.append(_)
	normalized_set.insert(0, feature_sets[0])
	print(normalized_set)



if __name__ == '__main__':
	main()