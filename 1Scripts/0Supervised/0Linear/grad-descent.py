#!/home/pratik/anaconda3/bin/python3.6

import sys
import config

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
def close_enough(current_theta, old_theta, tolerance):
	return all([ abs(i - j) < tolerance for i, j in zip(current_theta, old_theta)])

# this calculates the value of h_theta(x) as t0 + x1 * t1 + ... xn * tn
def h(thetas, xi_s, is_class = False):
	from math import nan, e
	try:
		p_sum = sum([i * j for i, j in zip(thetas, xi_s)])
	except TypeError:
		return nan
	if is_class:
		try:
			return 1 / (1 + e ** -p_sum)
		except OverflowError:
			return 1
	else:
		return p_sum

def logu(x):
	from math import log
	if x <= 0:
		return config.max_cost
	else: return log(x)

# this calculate the cost J(theta) for the given theta values
# good convergence test
def J(thetas, data, is_class, regular_param):
	from math import nan
	m = len(data)
	xi_s = [x[:-1] for x in data]
	yi_s = [x[-1] for x in data]
	regular_cost = regular_param / (2 * m) * sum(list(map(lambda x: x ** 2, thetas)))
	try:
		if is_class:
			p_sum = sum([-yi_s[_] * logu(h(thetas, xi_s[_], is_class)) \
				-(1 - yi_s[_]) * logu(1 - h(thetas, xi_s[_], is_class)) for _ in range(m)])
			return 1 / m * p_sum + regular_cost
		else:
			return 1 / (2 * m) * sum([(h(thetas, xi_s[_]) - yi_s[_]) ** 2 for _ in range(m)]) + regular_cost
	except OverflowError:
		return nan



# this is the core functionality that calculates the various theta values
def gradDesc(data_set, og_learning, n, m, timeout, tolerance, regular_param):
	# imports for the function
	from time import sleep
	from math import isnan, isinf
	from statistics import mean
	import datetime
	
	yi_s = [float(x[-1]) for x in data_set]
	is_class = all(list(map(lambda x: True if x in [0.0, 1.0, 1, 0] else False, yi_s)))
	ans = True
	learning_rate = og_learning
	rates_history =[]
	theta_history =[]
	run_count = 0
	if is_class:
		max_run = config.class_run_count
		max_iter = config.class_iter_count
	else:
		max_run = config.max_run_count
		max_iter = config.max_iterations

	while ans and run_count < max_run:
		# return theta for the current run
		theta = [ 0 for _ in range(n)]
		old_J = J(theta, data_set, is_class, regular_param)
		# cost_history = [tuple([0, old_J])]
		cost_history = []
		iter_count = 1
		check_flag = True
		# master break for long processes
		break_button = datetime.datetime.now()
		run_count += 1

		while True:
			current_theta = [ 0 for _ in range(n) ]
			for _ in range(n):
				partial_sum = sum([ (h(theta, i[:-1], is_class) - i[-1]) * i[_] for i in data_set])
				new_theta = theta[_] * (1 - regular_param / m) - (learning_rate / m) * partial_sum
				current_theta[_] = new_theta
			
			current_J = J(current_theta, data_set, is_class, regular_param)
			print(str(run_count) + ":" + str(iter_count), \
				" LR(α): ", learning_rate,end = " ")
			if regular_param:
				print("RR(λ): ", regular_param, end = "")
			print(" J(θ): ", current_J, "\nθ: ", current_theta, "\n")
			iter_count += 1
			old_rate = learning_rate
			
			# check for divergence
			if current_J > old_J:
				# (decrease learning rate)
				learning_rate /= config.J_divergence_factor
				print("THETA(S) HAS/HAVE DIVERGED", "RESULTS MAY BE INCORRECT", sep = "\n")
				check_flag = False
				break
			elif (old_J - current_J) / old_J < config.dyn_alpha_tolerance:
				rates_history.append(old_rate)
				theta_history.append(current_theta)
				learning_rate *= config.dyn_alpha_factor

			if iter_count > max_iter:
				learning_rate *= config.iter_overflow_factor
				check_flag = False
				break
			
			old_J = current_J
			cost_history.append(tuple([iter_count - 1, old_J]))

			# # divergence_test.append(all([ i > j for i, j in zip(current_theta, theta)]))
			# # if  all(divergence_test[-div_count:]) or \
			# if	all(list(map(lambda x: x > 10 ** 9, [abs(i - j) for i, j in zip(theta, current_theta)]))) or \
			#     all(list(map(lambda x: isnan(x) or isinf(x), theta))):
			# 	print("THETA(S) HAS/HAVE DIVERGED", "RESULTS WILL BE INCORRECT", sep = "\n")
			# 	break

			# check for convergence
			if close_enough(current_theta, theta, tolerance) or current_J < config.min_J_cost:
				# (increase learning rate)
				rates_history.append(old_rate)
				theta_history.append(current_theta)
				learning_rate *= config.better_theta_factor
				theta = current_theta
				break

			# check for time exceed  (increase learning rate)
			if (datetime.datetime.now() - break_button).seconds > timeout:
				print("TIME EXCEEDED, FLUSHING NOW", "RESULTS MAY BE INCORRECT", sep = "\n")
				learning_rate *= config.time_exceed_factor
				check_flag = False
				break
			
			theta = current_theta
		
		if not check_flag:
			cost_history = []
		
		print("* Learning Rate (α):", old_rate) 
		if check_flag or len(rates_history):
			ans = not (old_rate >= max(rates_history) and \
					run_count > config.min_run_count \
					and run_count < max_run)
		else:
			ans = True
		if ans:
			print("* Proposed Learning Rate (α):", learning_rate)

		# user input to continue below, uncomment if required
		# print("Try new learning_rate (Y/n) : ", end = "")
		# ans = str(input())
		# if ans == "" or ans == "y" or ans == "yes":
		# 	ans = True
		# else:
		# 	ans = False

		sleep(config.pause_theta)
	
	if not check_flag:
		theta = theta_history[rates_history.index(max(rates_history))]

	return theta, cost_history, is_class


# this is functionality that packages the various params required for gradient descent 
def calc_params(learning_rate, data_set, timeout, regular_param):
	normal = len(data_set['statistics']) != 0 # bool to check if data is normalized or not
	stats = data_set['statistics']
	learning_set = data_set['data_set'] # dataset that we will work with 
	yi_s = [float(x[-1]) for x in learning_set] # the values of each row
	n = len(learning_set[0]) - 1 # the number of features (including x0) ==> n - 1 actual features
	m = len(learning_set) # the number of data entries 
	return_obj = {
		'parameters' : {
			'thetas' : [],
			'cost_history' : []
		},
		'statistics' : [],
		'learning_rate' : learning_rate,
		'regularization_param' : regular_param,
		'n' : n,
		'm' : m,
		'is_class' : False
	}
	return_obj['parameters']['thetas'], return_obj['parameters']['cost_history'], return_obj['is_class'] \
	= gradDesc(learning_set, learning_rate, n, m, timeout, config.tolerance, regular_param)
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
def query_y(theta, n, stats = [], is_class = False):
	
	stats = [tuple([stats[i], stats[i + 1]]) for i in range(0, len(stats), 2)]
	
	# this is for taking input from the user
	print("Enter the xi(s)", end = " : ")
	xi_s = list(map(float, list(filter( lambda x: x != '', input().strip(' ').split(' ')))))[: n]
	xi_s.insert(0,1)

	if len(stats):
		xi_s = make_normal(xi_s, stats)
	value = h(theta, xi_s, is_class)
	if len(stats):
		print("<USING NORMALIZED DATA>", end = " ")
		value = (value * stats[-1][-1]) + stats[-1][0]
	if is_class:
		print("<P(Y=1|X;θ)>", end = ' ')
	print(value)
	return value

# this plots the cost function
def plot(jvalues, iter_counts, thetas):
	if len(jvalues):
		import matplotlib.pyplot as plt
		# if len(thetas) == 2:
		# 	from mpl_toolkits.mplot3d import Axes3D
		# 	fig = plt.figure()
		# 	ax = fig.gca(projection='3d')
		# 	ax.plot_trisurf([x[0] for x in thetas], y, z, linewidth = 0.2, antialiased = True)

		# else:
		plt.plot(iter_counts, jvalues, 'b-')
		plt.axis([min(iter_counts), max(iter_counts), min(jvalues), max(jvalues)])
		plt.axis('normal')
		plt.ylabel('J(θ) -> ')
		plt.xlabel('Iterations ->')
		# plt.title('Cost over time')
		plt.show()

# main program to tie everything together
def main():
	try:
		learning_rate = float(sys.argv[1])
		data_set = sys.argv[2]
		try:
			normalized_data = sys.argv[3]
			try:
				regular_param = float(sys.argv[4])
			except:
				regular_param = config.regularization_param
			try:
				timeout = float(sys.argv[5])
			except:
				timeout = config.timeout
		except:
			normalized_data = config.default_normal_filename
			timeout = config.timeout
	except:
		try:
			learning_rate, features, data_set, normalized_data = tuple(input().strip(' ').split(' '))
		except:
			normalized_data = ""

	data_set = process_data(data_set, normalized_data)
	params = calc_params(learning_rate, data_set, timeout, regular_param)
	cost_h = params['parameters']['cost_history']
	plot([x[-1] for x in cost_h], [x[0] for x in cost_h], params['parameters']['thetas'])
	
	answer = True
	while answer:
		query_y(params['parameters']['thetas'], params['n'], params['statistics'], params['is_class'])
		print("Calculate another (Y/n) : ", end = "")
		ans = str(input())
		answer = False
		if ans == "" or ans[0] == "" or ans[0] == "y":
			answer = True

# enables script access
if __name__ == '__main__':
	main()