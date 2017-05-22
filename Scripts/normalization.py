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

try:
	out_file = sys.argv[2]
except:
	out_file = "normalized_data.txt"

# this function removes any oddities in the data set
def sanitize(unset_data):
	return type(unset_data)(filter(lambda x: x != '', unset_data))

# this function extracts the feature-set from the data so that it can be normalized
def process(unset_data):
	m = len(unset_data)
	n = len(sanitize(unset_data[0].split(' '))) - 1
	feature_sets = [ [] for _ in range(n + 1) ]
	yi = []
	for _ in range(m):
		unset_data[_] = unset_data[_].split(' ')
		for i in range(n + 1):
			feature_sets[i].append(unset_data[_][i])
		yi.append(unset_data[_][-1])
	return tuple([feature_sets, yi])


# this function recontructs the dataset with the normalized data
'''
	the output file will be formatted as 
	u_i dev u_i dev ... u_i dev => n + 1 for each feature 
	
	x0 x1 x2 ... xn y1	+
	x0 x1 x2 ... xn y1	|
	.	.	.	.	.	|
	.	.	.	.	.	m rows
	.	.	.	.	.	|
	.	.	.	.	.	|
	x0 x1 x2 ... xn ym	+
	+----- n + 2 cols --+

'''
def reconstruct(normalized_set, yi_s):
	row = len(normalized_set[0])
	col = len(normalized_set)
	normalized_data = []
	for i in range(row):
		data_example = []
		for j in range(col):
			data_example.append(normalized_set[j][i])
		# data_example.append(yi_s[i])
		normalized_data.append(data_example)
	# print(normalized_data)
	return normalized_data

def file_write(filename, dataset):
	header = dataset[1]
	dataset = dataset[0]
	fout = open(filename, "w")
	for _ in header:
		fout.write(str(_[0]) + " " + str(_[1]) + " ")
	fout.write("\n")
	for _ in dataset:
		for i in _:
			fout.write(str(i) + " ")
		fout.write("\n")
	fout.close()



# this function pieces everything together
def normalize():
	unset_data = open(file_name).read().strip('\n').split('\n')
	feature_sets, yi_s = process(sanitize(unset_data))
	retrieve_data = []
	normalized_set = []
	for _ in feature_sets[1:]:
		_ = list(map(int, _))
		u_i = int(mean(_))
		dev = int(stdev(_))
		retrieve_data.append(tuple([u_i, dev]))
		_ = list(map(lambda x: (x - u_i) / dev, _))
		normalized_set.append(_)
	normalized_set.insert(0, feature_sets[0])
	retrieve_data.insert(0, tuple([0, 1]))
	normalized_data = reconstruct(normalized_set, yi_s)
	x = tuple([normalized_data, retrieve_data])
	file_write(out_file, x)
	return x


def main():
	x = normalize()
	# print(x)
	return x

if __name__ == '__main__':
	main()