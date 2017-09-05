#!/home/pratik/anaconda3/bin/python3.6
'''
	Neural network implementation vectorized with different activation functions
'''
from sys import argv as rd
import numpy as np

class Activations:

	def __init__(self):
		pass

	def sig(z):
		return 1 / (1 + np.exp(-z))

	def tanh(z):
		return np.tanh(z)

def activation_function(z, func = Activations.tanh):
	return func(z)
activation_function = np.vectorize(activation_function)

def process_data(filename, y):
	# set this according to how your data is organized
	split_char = ' '
	raw_data = open(filename).read().strip(' ').split('\n')
	raw_data = [ list(map(float, x.strip(' ').split(split_char))) for \
		x in list(filter(lambda x: x != '', raw_data)) ]
		
	X = np.array([_[:-y] for _ in raw_data]).T
	y = np.array([_[-y:] for _ in raw_data])
	return X, y

def init_weights(nodes):
	weights = []
	for _, i in enumerate(nodes[1:]):
		# replace random with specified weight generators
		# _ because enumerate still starts from 0
		w = np.random.rand(i, nodes[_])
		b = np.random.rand(w.shape[0], 1)
		weights.append(tuple([w, b]))
	return weights


def FP(X, weights):
	a_prev = X
	activations = [a_prev]
	for w_i, b_i in weights:
		assert(a_prev.shape[0] == w_i.shape[-1])
		z_current = np.dot(w_i, a_prev) + b_i
		a_current = activation_function(z_current)
		activations.append(a_current)
		a_prev = a_current
	# a_current.T === y is a column vector
	return activations, a_current.T

def main():
	filename = rd[1]
	layers = int(rd[2])
	nodes = []
	# follows the convention layers = hidden + output
	for _ in range(layers):
		print("Nodes in layer %d" % (_ + 1), end = " : ")
		try:
			x = int(input())
			nodes.append(x)
		except:
			layers -= 1

	X, y = process_data(filename, nodes[-1])
	nodes.insert(0, X.shape[0])
	weights = init_weights(nodes)
	activations, y_k = FP(X, weights)
	# print(np.ndarray.tolist(y_k))



if __name__ == '__main__':
	main()