'''

	This file contains all the utilitarian functions

'''

# sanitize list, tuple etc on the basis on a given value
def sanitize(data, value = ''):
	return type(data)(filter(lambda x: x != value, data))


# return an empty matrix of dimension m x n intialized to x
def matrix(m, n, x = []):
	if x == []:
		from random import random as r
		return [[r() for _ in range(n)] for i in range(m)]
	return [[x for _ in range(n)] for i in range(m)]

# flatten the theta list for easy summation of squares
def flatten(S):
    if empty(S):
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])

# this function is to separate out the stats and the data in case of normalized data
def process_data(data_file, normal_file, k):
	# this is the object that will be returned
	# in case of normalized data, the statistics portion will contain information to 
	# get back data from the normalized set
	return_obj = {
		'statistics' : [],
		'data_set' : [],
		'm' : 0
	}

	# always tries to parse the normalized data first
	try:
		normal_set = open(normal_file).read().strip(' ').strip('\n').split('\n')
		normal_set = list(map(lambda x: x.strip(' ').split(' '), normal_set))
		normal_set = [list(map(float, x)) for x in normal_set]
		return_obj['statistics'] = normal_set[0]
		return_obj['data_set'] = normal_set[1:]
	
	# parse usual data
	except IOError:
		data_set = open(data_file).read().strip(' ').strip('\n').split('\n')
		data_set = list(map(lambda x: x.strip(' ').split(' '), data_set))
		data_set = [list(map(float, x)) for x in data_set]
		return_obj['data_set'] = data_set
	except:
		return_obj['statistics'].append(-1)

	return_obj['m'] = len(return_obj['data_set'])
	
	# seperate out the k-dimensional output vector and the inputs for the m-set training array
	return_obj['data_set'] = [ [ _[:-k], _[-k:]] for _ in return_obj['data_set'] ]
	
	return return_obj['m'], return_obj['data_set'], return_obj['statistics']

# approx values
# 0 -> 0.5 +/- 36 -> +/- 1
def h(thetas, xi_s):
	from math import e
	p_sum = sum([i * j for i, j in zip(thetas, xi_s)])
	return 1 / (1 + e ** -p_sum)