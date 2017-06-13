#!/home/pratik/anaconda3/bin/python3.6

'''
	This is an implementation of a fixed neural network using backpropogation 
	(It calculates the xor of variables using one output node and two hidden layers)

'''

# imports 
import config
import sys



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
	nodes_per = []
	

	for _ in range(2, hidden + 2):
		print("Nodes in layer", _, end = " : ")
		nodes_per.append(int(input()))
	nodes_per.append(k)

	m, dataset, statistics = process_data(data_file, normal_file, k)
	nodes_per.insert(0, len(dataset[0][0])) # the number of inputs
	print(nodes_per)
	pass

if __name__ == '__main__':
	print("Neural Network small scale fixed implementation (v1.0)")
	main()
