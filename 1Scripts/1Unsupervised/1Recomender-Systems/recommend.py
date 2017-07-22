#!/home/pratik/anaconda3/bin/python3.6

"""
	This script takes a dataset and sets up a recommender system that can 
	suggest values according to data given.
	Data should be of the form 
	n_m x n_u -> 
	n_u -> N users and recommendations according to the users
	n_m -> m samples of correlated items
	NOTE : DEFAULT VALUE FOR EMPTY DATA IS 0
	(change this is config, if required)

"""
# default imports
import numpy as np
import matplotlib.pyplot as plt
from sys import argv as rd
from sys import stdout
import helper
import config

# custom imports
from time import sleep

# returns the initial xi_s and thetas
# xi_s -> 	n_m x n
# thetas -> n_u x n
def initialize(n, n_m, n_u):
	return (-1 + 2 * np.random.random((n_m, n))),\
		(-1 + 2 * np.random.random((n_u, n)))


# returns a list of triples as (i, j, y1), (i, j, y2)...
def getFilled(dataset):
	truth_mat = np.ndarray.tolist(dataset != config.e_val)
	fill_array = []
	for i, row in enumerate(truth_mat):
		for j, y in enumerate(row):
			if dataset[i,j]:
				fill_array.append((i, j, dataset[i, j]))
	return fill_array

				
# Cost function J
def J(xi_s, thetas, regular = False):
	from time import sleep
	theta_mat, xi_mat, y_mat = [], [], np.array([_[-1] for _ in fill_array])
	for i, j, y in fill_array:
		theta_mat.append(thetas[j]), xi_mat.append(xi_s[i])
	theta_mat, xi_mat = np.array(theta_mat), np.array(xi_mat)
	cost = np.sum((np.sum(theta_mat * xi_mat, 1) - y_mat) ** 2) / 2
	# if regularized
	if regular:
		cost += (np.sum(theta_mat ** 2) + np.sum(xi_mat ** 2)) / regular
	return cost

# function that calculates the final feature and theta values
# keeping only cost as an esacpe
def grad_descent(n_m, n_u, n, learning_rate, regular = False):
	looks = ["-","\\", "|", "/"]
	init_xi_s, init_thetas = initialize(n, n_m, n_u)

	# change over the loop
	current_xi_s = init_xi_s
	current_thetas = init_thetas

	# logging history
	cost_history = [J(init_xi_s, init_thetas, regular)]
	theta_history = [current_thetas]
	xi_history = [current_xi_s]

	# used to carry out the update of xi_s
	x_update = {}
	for i, j, y in fill_array:
		try:
			x_update[i].append((j, y))
		except KeyError:
			x_update[i] = [(j, y)]

	# used to carry out the update of thetas
	theta_update = {}
	for i, j, y in fill_array:
		try:
			theta_update[j].append((i, y))
		except KeyError:
			theta_update[j] = [(i, y)]
	
	run_count = 1
	while True:
		stdout.write("%s #%d L.R (α) : %f J(X, Θ) : %f\r" % \
			(looks[run_count % 4], run_count, learning_rate, cost_history[-1]))
		# stdout.write("[%s]\r" %((run_count % 10 == 0) * int(run_count / 10) * "█"))
		stdout.flush()
		
		new_x = np.zeros(current_xi_s.shape)
		for i in x_update:
			cost = np.zeros((1, n))
			# adding the relative errors for gradient
			for j, y in x_update[i]:
				cost += ((np.sum(np.multiply(current_thetas[j],\
					current_xi_s[i])) - y) * current_thetas[j])
			# adding regular term
			if regular:
				cost += regular * current_xi_s[i]
			new_x[i] = current_xi_s[i] - (learning_rate * cost)

		new_thetas = np.zeros(current_thetas.shape)
		for j in theta_update:
			cost = np.zeros((1, n))
			# adding the relative errors
			for i, y in theta_update[j]:
				cost += ((np.sum(np.multiply(current_thetas[j],\
					current_xi_s[i])) - y) * current_xi_s[i])
			# regularization param
			if regular:
				cost += regular * current_thetas[j]
			new_thetas[j] = current_thetas[j] - (learning_rate * cost)
		
		assert(new_x.shape == current_xi_s.shape)
		assert(new_thetas.shape == current_thetas.shape)
		
		# updating the history to add new params
		new_cost = J(new_x, new_thetas, regular)
		
		run_count += 1

		# break checks for convergence/ divergence
		if new_cost > cost_history[-1]:
			print("\nJ Divergence!\n")
			learning_rate /= 7
			continue

		elif abs(new_cost - cost_history[-1]) < config.J_meet:
			print("\nJ has converged!\n")
			break

		elif run_count > config.min_run_count and \
			helper.close_enough(new_cost, cost_history[-1]):
			print("\nJ converging!")
			learning_rate *= 3

		if run_count > config.max_run_count:
			print("\nRUN OVERFLOW\n")
			break
		
		theta_history.append(new_thetas)
		xi_history.append(new_x)
		cost_history.append(new_cost)

		# preparing for next iteration
		current_xi_s = new_x
		current_thetas = new_thetas

	# returning back learnt params
	final_xi_s = xi_history[-1]
	final_thetas = theta_history[-1]
	return final_xi_s, final_thetas

# function to fill in the missing data values
def fill_data(dataset, xi_s, thetas):
	n_m, n_u = dataset.shape
	new_data = dataset
	for i in range(n_m):
		for j in range(n_u):
			if dataset[i,j] == config.e_val:
				new_data[i,j] = np.sum(np.multiply(thetas[j], xi_s[i]))
	return new_data


# tieing everything together
def main():
	filename = rd[1]
	# no of features to represent each "thing" 
	# more is better, but more expensive
	n = int(rd[2])
	learning_rate = float(rd[3]) 
	# regularized
	try:
		regular = float(rd[4])
	except:
		# regular = False
		regular = config.regular

	dataset = np.matrix(helper.process_data(filename))
	
	# this contains the i,j and value of the filled elements
	global fill_array
	fill_array = getFilled(dataset)
	n_m, n_u = dataset.shape
	# learn the defining features as well as preferences for users
	final_xi_s, final_thetas = grad_descent(n_m, n_u, n, learning_rate, regular)
	print("FINAL THETAS", final_thetas, "FINAL XIS", final_xi_s, sep = "\n")
	# fill in the approx blanks now that params have been learnt
	new_dataset = fill_data(dataset, final_xi_s, final_thetas)


# script access
if __name__ == '__main__':
	print("Recommender System (v1.4)")
	main()