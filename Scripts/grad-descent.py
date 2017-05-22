#!/usr/bin/python3

import sys

'''

	This program predicts data on the basis of an input set and determines the best learning rate and 
	other features automatically. You have to give it the number of features that will be implemented.

'''

# this function is to separate out the stats and the data in case of normalized data
def process_data(data_file, normal_file):
	# this is the object that will be returned
	# in case of normalized data, the statistics portion will contain information to 
	# get back data from the normalized set
	return_obj = {
		'statistics' : [],
		'data_set' : []
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
		data_set = [list(map(int, x)) for x in data_set]
		return_obj['data_set'] = data_set
	except:
		return_obj['statistics'].append(-1)
	return return_obj

# checks for convergence of values
def close_enough(current_theta, old_theta, tolerance = 0.001):
	return all([ abs(i - j) < tolerance for i, j in zip(current_theta, old_theta)])

# this calculates the value of h_theta(x) as t0 + x1 * t1 + ... xn * tn
def h(thetas, xi_s):
	from math import nan
	try:
		return sum([i * j for i, j in zip(thetas, xi_s)])
	except TypeError:
		return nan

# this is the core functionality that calculates the various theta values
def gradDesc(data_set, learning_rate, n, m, timeout = 30, tolerance = 0.000001):
	# imports for the function
	from math import isnan, isinf
	import datetime

	# return theta for the current run
	theta = [ 0 for _ in range(n)]
	iter_count = 1
	# divergence_test = []
	# div_count = 5

	# master break for long processes
	break_button = datetime.datetime.now()

	while True:
		current_theta = [ 0 for _ in range(n) ]
		for _ in range(n):
			partial_sum = sum([ (h(theta, i[:-1]) - i[-1]) * i[_] for i in data_set])
			new_theta = theta[_] - (learning_rate / m) * partial_sum
			current_theta[_] = new_theta
		print(iter_count, current_theta)
		iter_count += 1
		
		# check for divergence
		
		# divergence_test.append(all([ i > j for i, j in zip(current_theta, theta)]))
		# if len(divergence_test) > div_count:
		# 	divergence_test = divergence_test[-div_count :]
		# print(divergence_test)
		# if  all(divergence_test[-div_count:]) or \
		if	all(list(map(lambda x: x > 10 ** 9, [abs(i - j) for i, j in zip(theta, current_theta)]))) or \
		    all(list(map(lambda x: isnan(x) or isinf(x), theta))):
			print("THETA(S) HAS/HAVE DIVERGED", "RESULTS WILL BE INCORRECT", sep = "\n")
			break

		# check for convergence
		if close_enough(current_theta, theta, tolerance):
			theta = current_theta
			break

		# check for time exceed
		if (datetime.datetime.now() - break_button).seconds > timeout:
			print("TIME EXCEEDED, FLUSHING NOW", "RESULTS WILL BE INCORRECT", sep = "\n")
			break
		
		theta = current_theta
	return theta


# this is functionality that packages the various params required for gradient descent 
def calc_params(learning_rate, data_set, timeout = 30):
	normal = len(data_set['statistics']) != 0 # bool to check if data is normalized or not
	stats = data_set['statistics']
	learning_set = data_set['data_set'] # dataset that we will work with 
	yi_s = [float(x[-1]) for x in learning_set] # the values of each row
	n = len(learning_set[0]) - 1 # the number of features (including x0) ==> n - 1 actual features
	m = len(learning_set) # the number of data entries 
	return_obj = {
		'parameters' : [],
		'statistics' : [],
		'learning_rate' : learning_rate,
		'n' : n,
		'm' : m,
	}
	return_obj['parameters'] = gradDesc(learning_set, learning_rate, n, m, timeout)
	if normal:
		return_obj['statistics'] = stats
	return return_obj

# to normalize the input data if required
def make_normal(xi_s, stats):
	if len(xi_s) == len(stats) - 1:
		return [ (i - j[0]) / j[-1] for i, j in zip(xi_s, stats)]
	else:
		return None


# once query result and normalize xi_s if required
def query_y(theta, n, stats = []):
	
	stats = [tuple([stats[i], stats[i + 1]]) for i in range(0, len(stats), 2)]
	
	# this is for taking input from the user
	print("Enter the xi(s)", end = " : ")
	xi_s = list(map(float, list(filter( lambda x: x != '', input().strip(' ').split(' ')))))[: n]

	# use next three lines for checking the accuracy of the program
	# xi_s = [1234]
	# ./grad-descent.py 0.0099999 dataset2.txt dataset2_normal.txt => normalized
	# ./grad-descent.py 0.0000000769 dataset2.txt => unormalized 

	xi_s.insert(0,1)
	if len(stats):
		xi_s = make_normal(xi_s, stats)
	value = h(theta, xi_s)
	if len(stats):
		print("<USING NORMALIZED DATA>", end = " ")
		value = (value * stats[-1][-1]) + stats[-1][0]
	print(value)
	return value

# main program to tie everything together
def main():
	try:
		learning_rate = float(sys.argv[1])
		data_set = sys.argv[2]
		try:
			normalized_data = sys.argv[3]
			try:
				timeout = float(sys.argv[4])
			except:
				timeout = 30
		except:
			normalized_data = "normal_data.txt"
			timeout = 30
	except:
		try:
			learning_rate, features, data_set, normalized_data = tuple(input().strip(' ').split(' '))
		except:
			normalized_data = ""

	data_set = process_data(data_set, normalized_data)
	params = calc_params(learning_rate, data_set, timeout)

	answer = True

	while answer:
		query_y(params['parameters'], params['n'], params['statistics'])
		print("Calculate another (Y/n) : ", end = "")
		ans = str(input())
		answer = False
		if ans == "" or ans[0] == "" or ans[0] == "y":
			answer = True

# enables script access
if __name__ == '__main__':
	main()