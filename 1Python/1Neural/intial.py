#!/home/pratik/anaconda3/bin/python3.6

'''
	This is an implementation of a fixed neural network using backpropogation 
	(It calculates the xor of variables using one output node and two hidden layers)

'''

# default imports 
import config
import helper

# specific imports
import sys
# from time import sleep

# function for making the theta matrix based on the network config
def make_theta(nodes_per):
	nodes_per = list(map(lambda x: x + 1, nodes_per))
	thetas =[helper.matrix(i, j) for i, j in zip(nodes_per[1:], nodes_per)]
	thetas.insert(0, []) # offsetting the thetas so as to keep with labeling
	return thetas

# propogate one training example through the network 
# with the giving set of theta values
# basically passes through all the layers ===> equiv to one run of the forward alg.
def FP(x, thetas, example_no = []):
	print("x(", example_no, ") :", x[0])
	activation_history = []

	# offsetting the history to keep in norm with the labeling
	activation_history.append([])
	
	# adding inital bias unit
	x[0].insert(0, 1)

	# inital seperating
	yi_s = x[-1]
	x = x[0]
	activation_history.append(x)
	count_layer = 1
	for i in thetas[1:]:
		print("L", count_layer, " -> ", "L", count_layer + 1, sep = '')
		print("Theta set : ", i)
		temp_ans = []
		# iterating through the 
		count_node = 0
		for j in i:
			# skip bias unit calculation (effective speed up)
			if not count_node:
				count_node += 1
				continue
			print("Node", count_node)
			# adding activation unit 
			temp_ans.append(helper.h(j, x))
			print("Thetas : ", j)
			print("a(", count_layer + 1, ",", count_node, ") -> ", temp_ans[-1], sep = "")
			count_node += 1

		# adding the bias unit for the next run
		temp_ans.insert(0, 1)
		x = temp_ans
		activation_history.append(x)
		count_layer += 1
	
	print("Neural O/P : ", activation_history[-1][1:])
	print("y(", example_no, ") : ", yi_s)

	return activation_history


# for delta calculation per training set example
def BP(x, thetas):
	pass


def main():
	# number of hidden layers
	hidden = int(sys.argv[1])
	L = hidden + 2 # L is the total no of layers
	data_file = sys.argv[2] #dataset filename
	try:
		k = int(sys.argv[3]) # no of output nodes, ie, nodes in the Lth layers
	except:
		k = config.k
	try:
		normal_file = sys.argv[4] # normalized datset name
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
	thetas = make_theta(nodes_per)

	count_example = 1
	for t_ex in dataset:
		print("Training ex", count_example)
		history = FP(t_ex, thetas, count_example)
		print("x", count_example, "Activation matrix : ", history)
		BP(t_ex, thetas)
		
		count_example += 1
		print("Press ENTER to continue... ", end = "")
		input()

if __name__ == '__main__':
	print("Neural Network dynamic implementation (v1.3)")
	main()
