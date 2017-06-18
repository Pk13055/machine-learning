#!/home/pratik/anaconda3/bin/python3.6


''' 
	# numpy implementation for greater speed
	this is a numpy based implementation of an NN, for 
	greater speed and accuracy
	# every function is expected to return a python dict return object 
	# for easier readability and debugging
'''

# default imports
import helper
import config
import numpy as np

# specific imports
import sys
import operator
from copy import deepcopy
from math import sqrt, e

r6 = sqrt(6)

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
		'empty_DIJ' : []
	}

	# initializing empty theta and DIJ
	thetas = [np.array([0])]
	empty_DIJ = [np.array([0])]

	# dimensions for the theta(l) matrices
	dimen_mat = [[y, x + 1] for x, y in zip(nodes, nodes[1:])]
	return_obj['dimensions'] = dimen_mat
	return_obj['L'] = len(dimen_mat)

	for i in dimen_mat:
		epsilon = r6 / sqrt(i[0] + (i[-1] - 1))
		thetas.append(-epsilon + (2 * epsilon) * np.random.rand(i[0], i[-1]))
		empty_DIJ.append(np.zeros((i[0], i[-1])))

	return_obj['thetas'] = thetas
	return_obj['empty_DIJ'] = empty_DIJ
	return helper.unpack(return_obj)

# calculates gradients with regularization
def calc_gradient(thetas, raw_data, empty_DIJ, lambd = 0):
	
	return_obj = {
		'Dij' : [],
		'lambda' : lambd,
	}

	# copying the dataset to prevent change
	dataset = deepcopy(raw_data)
	Dij = empty_DIJ
	
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
			activation_layers.append(current_activated)
			
			# adding bias unit 
			next_layer = np.r_[[[1]], current_activated]

			# for the next iteration after adding bias unit
			current_activation = next_layer

		# error associated with the output
		delta_L = activation_layers[-1] - yi
		
		# these deltas are required every iteration
		delta_layers = [delta_L]

		# setting delta for current iteration
		previous_delta = delta_L

		# L required for certain index calc.
		L = len(activation_layers) - 1

		# performing backpropogation
		for idx, l in enumerate(activation_layers[-2::-1]):
			act_req = l[:, 2:]
			
			# calculating current layer delta
			print("DELTA %d" % (L - (idx + 1)))
			print(thetas[L - (idx + 1)][:, 2:].T)
			print("ADAS")
			try:
				current_delta = ((thetas[L - (idx + 1)][:, 2:]).T).dot(previous_delta) 
					# (act_req * (1 - act_req))
			except:
				print("FAIL")
				print("THETA", thetas[L - (idx + 1)][:, 2:].T)
				print("DELTA", previous_delta)
				input()
			
			# adding the delta layer to the legend
			delta_layers.append(current_delta)
			
			# continuing for next iteration
			previous_delta = current_delta



	return_obj['Dij'] = Dij
	return helper.unpack(return_obj)


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
	nodes_per.append(k)
	nodes_per.insert(0, len(dataset[0][0])) # the number of inputs
	L, nodes_per, inital_thetas, empty_DIJ = design_thetas(nodes_per)

	partial_D, lambd, activations = calc_gradient(inital_thetas, dataset, empty_DIJ, lambd)
	print(partial_D)

if __name__ == '__main__':
	main()