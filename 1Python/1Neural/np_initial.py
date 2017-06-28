#!/home/pratik/anaconda3/bin/python3.6


''' 
	# numpy implementation for greater speed
	this is a numpy based implementation of an NN, for 
	greater speed and accuracy
	# every function is expected to return a python dict return object 
	# for easier readability and debugging
'''

# default imports
import numpy as np
import helper
import np_helper
import config

# specific imports
import sys
import operator
from math import sqrt, e, inf
from time import sleep


# this creates a function that can be applied to matrices element wise
def sigmoid(z):
	return 1 / (1 + e ** -z)
sigmoid = np.vectorize(sigmoid)

# initializes the thetas 'less' randomly
def design_thetas(nodes):

	return_obj = {
		'L' : 0,
		'dimensions' : [],
		'thetas' : [],
		# 'empty_DIJ' : []
	}

	r6 = sqrt(6)
	# initializing empty theta and DIJ
	thetas = [np.array([0])]
	# empty_DIJ = [np.array([0])]

	# dimensions for the theta(l) matrices
	dimen_mat = [[y, x + 1] for x, y in zip(nodes, nodes[1:])]
	return_obj['dimensions'] = dimen_mat
	return_obj['L'] = len(dimen_mat)

	for i in dimen_mat:
		epsilon = r6 / sqrt(i[0] + (i[-1] - 1))
		thetas.append(-epsilon + (2 * epsilon) * np.random.rand(i[0], i[-1]))
		# empty_DIJ.append(np.zeros((i[0], i[-1])))

	return_obj['thetas'] = thetas
	# return_obj['empty_DIJ'] = empty_DIJ
	return helper.unpack(return_obj)


# calculates gradients with regularization
def calc_gradient(thetas, raw_data, lambd = 0):
	
	return_obj = {
		'Dij' : [],
		'cost' : 0,
		'lambda' : lambd
	}

	# copying the dataset to prevent change
	# initalizing useful params
	dataset = raw_data
	Dij = []
	is_empty = True
	cost = 0
	m = len(dataset)

	# iterating over the dataset, 
	# performing FP and BP
	for i in dataset:
		
		# these layers are used to prevent redundant calc for BP
		activation_layers = [[]]
		
		# inital layer taken from the dataset
		initial_layer = np.matrix([1] + i[0])
		yi = np.matrix(i[-1]).T # expected output
		
		# inital activation value set to input data
		current_activation = initial_layer.T
		
		# adding inital activation a(1) -> input
		activation_layers.append(current_activation)
		
		# iterating over the layers performing FP
		for theta_L in thetas[1:]:
			
			# creating raw non-activated
			current_raw = theta_L.dot(current_activation)
			
			# sigmoiding the previous output prior to adding one
			current_activated = sigmoid(current_raw)
			
			# adding bias unit 
			next_layer = np.r_[[[1]], current_activated]
			activation_layers.append(next_layer)

			# for the next iteration after adding bias unit
			current_activation = next_layer

		# removing final layer bias unit
		activation_layers[-1] = activation_layers[-1][1:, :]
		
		# error associated with the output
		delta_L = activation_layers[-1] - yi
		
		# these deltas are required every iteration
		delta_layers = [delta_L]

		# setting delta for current iteration
		previous_delta = delta_L

		# L required for certain index calc.
		L = len(activation_layers) - 1

		# accumilating the cost cost
		cost += (np.multiply(yi, np.log(activation_layers[-1])) + \
			np.multiply((1 - yi), np.log(1 - activation_layers[-1])))

		# performing backpropogation
		for idx, l in enumerate(activation_layers[-2:1:-1]):
			
			# calculating current layer delta
			current_delta = ((thetas[L - (idx + 1)][:, 1:]).T).dot(previous_delta)
			active_factor =  np.multiply(l[1:,:] , (1 - l[1:, :]))
			current_delta = np.multiply(current_delta, active_factor)
			# print("D%d" % (L - (idx + 1)), current_delta)


			# adding the delta layer to the legend
			delta_layers.append(current_delta)
			
			# continuing for next iteration
			previous_delta = current_delta

		# accumilating calculated delta for all layers

		# setting up the activation layers 

		activation_layers = activation_layers[1:-1] # removing final activation layer
		activation_layers = activation_layers[::-1] #reversing list to have a(L - 1) a(L - 2) ... a1
		# deltas will be delta(L) delta(L - 1) ... D2

		# partial derivative calculated for the current training example
		current_Dij = [del_current_plus_one * np.matrix(active_current).T \
			for del_current_plus_one, active_current in zip(delta_layers, activation_layers)]

		# accumilating the partial derivative over the training examples
		Dij = [ a + b for a, b in zip(Dij, current_Dij)]
		# initializes the Dij matrix with the first calculated cost if empty
		if is_empty:
			Dij = current_Dij
			is_empty = False

	# adding regularization and averaging partials
	
	Dij = [ (1 / m) * x for x in Dij][::-1]
	cost = (-1 / m) * cost[0, 0]
	
	if lambd:
		# regularizing cost cost 
		# summing the thetas .^ 2 except theta_0
		regular_cost = (lambd / (2 * m)) * sum([np.sum(np.multiply(y[:, 1:], y[:, 1:])) for y in thetas[1:]])
		cost += regular_cost

		# regularizing the partial derivatives except D0
		regular_factor = (lambd / m)
		regular_matrix = [ np.matrix([[regular_factor * x if j else 0 for j, x in enumerate(row)] \
			for row in mat]) for mat in thetas[1:] ]
		Dij = [ a + b for a, b in zip(Dij, regular_matrix)]
		# print("R", regular_matrix, "d", Dij, "t", thetas, sep = "\n")
	
	return_obj['cost'] = cost
	return_obj['Dij'] = Dij
	return helper.unpack(return_obj)


# this function does the main tieing up, computes the final thetas on the basis of a number of factors
def learn_thetas(initial_thetas, dataset, learning_rate, lambd):

	theta_history = [initial_thetas] # stores the history of thetas for convergence check
	J_history = [inf]
	current_thetas = initial_thetas # thetas used during every iteration
	run_count = 0 # keeps track of the iterations 
	no_error = True
	not_reset = True
	
	return_obj = {
		# "learnt_thetas" : [],
		"thetas" : []
		# "run_count" : 0,
		# "learning_rate" : 0,
		# "regularization_param" : 0
	}

	# infinite loop which will be broken by certain divergence and convergence checks
	while True:
		
		# partial derivative are calculated as gradients (with regularization) and returned here
		partial_D, lambd, cost = calc_gradient(current_thetas, dataset, lambd)
		J_history.append(cost)
		
		# theta* = theta - alpha * partial_d
		partial_D = [ -learning_rate * x for x in partial_D]
		new_thetas = [np.matrix([0])] + [ np.matrix(a) + b for a, b in zip(current_thetas[1:], partial_D)]
		theta_history.append(new_thetas)
		
		print("#%d : " % run_count, "L(α) :", learning_rate, "J(Θ) :", cost)
		# helper.print_thetas(new_thetas)
		
		# divergence / convergence conditions

		# cost is way too high
		if J_history[-1] > config.J_max:
			print("J Divergence")
			no_error = False
			break
		
		# Cost increased
		elif J_history[-1] > J_history[-2]:
			print("\nJ divergence\n")
			print("New learning rate (α) : ", learning_rate)
			sleep(1)
			learning_rate /= 3
			# break

		# cost is almost constant
		elif abs(J_history[-1] - J_history[-2]) / abs(J_history[-1]) < config.tolerance:			
			no_error = True
			break
		
		# J is decreasing too slowly
		elif run_count > config.min_run_count and \
			abs(J_history[-1] - J_history[-2]) < config.J_tolerance:
			learning_rate *= 1.05

		# thetas converged
		if run_count > config.max_run_count / 2:
			if not_reset:
				theta_history = theta_history[len(theta_history) / 2 :]
				not_reset = False
			if np_helper.close_theta(theta_history[-1], theta_history[-2], config.theta_tolerance):
				print("\nTheta Convergence!\n")
				no_error = True
				break

		# overflowed the run count
		if run_count > config.max_run_count:
			print("\nRun overflow\n")
			no_error = False
			break

		# readying for next iteration
		current_thetas = new_thetas
		run_count += 1

	# fill in details and return
	return_obj['thetas'] = theta_history
	print("\n Final Thetaset -> ", theta_history[-1][1:], "\n")
	# return_obj['learn_thetas'] = theta_history[-1]
	# return_obj['run_count'] = run_count  
	# return_obj['learning_rate'] = learning_rate
	# return_obj['regularization_param'] = lambd
	return theta_history
	# return helper.unpack(return_obj)


# after learning of params, passing new parameters
def query_y(thetas):
	
	# this is for taking input from the user
	print("Enter the xi(s)", end = " : ")
	xi_s = list(map(float, list(filter( lambda x: x != '', input().strip(' ').split(' ')))))
	xi_s.insert(0,1)
	valid_input = True
	print("I/P (x) -> ", xi_s[1:])
	
	# generates output
	# inital activation value set to input data
	current_activation = np.matrix(xi_s).T
	
	# adding inital activation a(1) -> input
	activation_layers = []
	activation_layers.append(current_activation)
	
	# iterating over the layers performing FP
	for theta_L in thetas[1:]:
		
		# creating raw non-activated
		try:
			current_raw = theta_L.dot(current_activation)
		except:
			valid_input = False
			break
		
		# sigmoiding the previous output prior to adding one
		current_activated = sigmoid(current_raw)
		
		# adding bias unit 
		next_layer = np.r_[[[1]], current_activated]
		activation_layers.append(next_layer)

		# for the next iteration after adding bias unit
		current_activation = next_layer

	if valid_input:
		y_matrix = activation_layers[-1][1:]
	else:
		y_matrix = ["Invalid Input"]
	
	print("O/P (y) -> ", y_matrix)
	return y_matrix

# taking params and initializing
def main():
	try:
		learning_rate = float(sys.argv[1])
	except:
		learning_rate = config.learning_rate
	# number of hidden layers
	hidden = int(sys.argv[2])
	L = hidden + 2 # L is the total no of layers
	data_file = sys.argv[3] #dataset filename
	
	# regularization parameter
	try:
		lambd = float(sys.argv[4])
	except:
		lambd = config.lambd
	# layers
	try:
		k = int(sys.argv[5]) # no of output nodes, ie, nodes in the Lth layers
	except:
		k = config.k
	try:
		normal_file = sys.argv[6] # normalized datset name
	except:
		normal_file = config.normal_file

	# the number of nodes per layers excluding the biasing unit
	# this will be used to build the theta array 
	nodes_per = []

	for _ in range(2, hidden + 2):
		print("Nodes in layer", _, end = " : ")
		nodes_per.append(int(input()))
	
	m, dataset, statistics = helper.process_data(data_file, normal_file, k)
	print(dataset)
	input()
	nodes_per.append(k)
	nodes_per.insert(0, len(dataset[0][0])) # the number of inputs
	L, nodes_per, inital_thetas = design_thetas(nodes_per)

	# final_thetas, theta_history, total_runs, final_rate, regular_param \
	# = learn_thetas(inital_thetas, dataset, learning_rate, lambd)
	theta_history = learn_thetas(inital_thetas, dataset, learning_rate, lambd)
	final_thetas = theta_history[-1]

	cont = True
	while cont:
		query_y(final_thetas)
		print("Calculate another (Y/n) : ", end = "")
		ans, cont = str(input()), False
		if ans == "" or ans[0] == "" or ans[0] == "y":
			cont = True


# script access
if __name__ == '__main__':
	main()