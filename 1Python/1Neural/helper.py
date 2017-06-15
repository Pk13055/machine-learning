'''

	This file contains all the utilitarian functions

'''
import operator

# sanitize list, tuple etc on the basis on l10 given value
def sanitize(data, value = ''):
	return type(data)(filter(lambda x: x != value, data))


# return an empty matrix of dimension m x n intialized to x
def matrix(m, n, x = []):
	if x == []:
		from random import random as r
		return [[0 for x in range(n)] if i == 0 else [r() for _ in range(n)] for i in range(m)]
	return [[0 for x in range(n)] if i == 0 else [x for _ in range(n)] for i in range(m)]

# flatten the theta list for easy summation of squares
def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])

# this function adds recursively nested lists element-wise
def list_recur(l1, l2, op = operator.add):
	if not l1:
		return type(l1)([])
	elif isinstance(l1[0], type(l1)):
		return type(l1)([list_recur(l1[0], l2[0], op)]) + list_recur(l1[1:],l2[1:], op)
	else:
		return type(l1)([op(l1[0], l2[0])]) + list_recur(l1[1:], l2[1:], op)

# checks for closeness, convergence of theta values
def close_enough(new_theta, old_theta, tolerance = 0.000001):
	return all(flatten(list_recur(new_theta, old_theta, lambda x, y : abs(x - y) < tolerance)))

# this function checks for divergence of J(theta)
def check_divergence(new_D, old_D):
	return any(flatten(list_recur(new_D, old_D, lambda x, y : x < y)))

# this function elementwise operators
def nest_operate(a, l, op = operator.add):
	if not l:
		return []
	elif isinstance(l[0], list):
		return [nest_operate(a, l[0], op)] + nest_operate(a, l[1:], op)
	else:
		return [op(a, l[0])] + nest_operate(a, l[1:], op)


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

# defined for making own operators
class Infix:
    def __init__(self, function):
        self.function = function
    def __ror__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))
    def __or__(self, other):
        return self.function(other)
    def __rlshift__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))
    def __rshift__(self, other):
        return self.function(other)
    def __call__(self, value1, value2):
        return self.function(value1, value2)