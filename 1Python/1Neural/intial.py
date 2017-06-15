#!/home/pratik/anaconda3/bin/python3.6

'''
	This is an implementation of a fixed neural network using backpropogation 
	(It calculates the xor of variables using one output node and two hidden layers)
	This script utilizes a lot of print statments so that the output of the network can be easily
	piped to a text document for later referral on it's working. 
	Output is tried to be made as graphical without being graphical 

'''

# default imports 
import config
import helper

# specific imports
import sys
import operator
# from time import sleep

# function for making the theta matrix based on the network config
def make_theta(nodes_per, val = None):
	nodes_per = list(map(lambda x: x + 1, nodes_per))
	if val is not None:
		thetas =[helper.matrix(i, j, val) for i, j in zip(nodes_per[1:], nodes_per)]
	else:
		thetas =[helper.matrix(i, j) for i, j in zip(nodes_per[1:], nodes_per)]
	thetas.insert(0, []) # offsetting the thetas so as to keep with labeling
	return thetas

# propogate one training example through the network 
# with the giving set of theta values
# basically passes through all the layers ===> equiv to one run of the forward alg.
def FP(x1, thetas, example_no = [], singleton = False):
	x = x1
	# print("x(", example_no, ") :", x[0])
	activation_history = []
	# offsetting the history to keep in norm with the labeling
	activation_history.append([])
	

	# adding inital bias unit
	if singleton:
		x.insert(0, 1)
	else:
		x[0].insert(0, 1)

	# inital seperating
	if not singleton:
		yi_s = x[-1]
		x = x[0]

	activation_history.append(x)
	count_layer = 1
	for i in thetas[1:]:
		# print("<====== L", count_layer, " -> ", "L", count_layer + 1, " =====>", sep = '')
		# print("Theta set : ", i)
		temp_ans = []
		# iterating through the 
		count_node = 0
		for j in i:
			# skip bias unit calculation (effective speed up)
			if not count_node:
				count_node += 1
				continue
			# print("Node", count_node)
			# adding activation unit 
			temp_ans.append(helper.h(j, x))
			# print("Thetas : ", j)
			# print("a(", count_layer + 1, ",", count_node, ") -> ", temp_ans[-1], sep = "")
			count_node += 1

		# adding the bias unit for the next run
		temp_ans.insert(0, 1)
		x = temp_ans
		activation_history.append(x)
		count_layer += 1
		# print("<================>")

	# print("****** Final Remarks ******")
	# print("Neural O/P : ", activation_history[-1][1:])
	# print("y(", example_no, ") : ", yi_s)
	if singleton:
		return activation_history[-1][1:]
	return activation_history, [i - j for i, j in zip(activation_history[-1][1:], yi_s)]


# for delta calculation per training set example
def BP(delta_L, thetas, nodes_pe, activation_history, L = 0):
	# nodes_per = list(map(lambda x: x + 1, nodes_per))
	nodes_per = [0] + nodes_pe
	if L == 0:
		L = len(thetas)

	master_delta = []
	# conciliating all the deltas
	next_delta = delta_L
	next_delta.insert(0, 0)
	master_delta.append(next_delta)

	for l in range(L - 1, 1, -1):
		# print("<====== L", l, " <- ", "L", l + 1, " =====>", sep = '')
		current_delta = []
		current_nodes = nodes_per[l] 
		next_nodes = nodes_per[l + 1] 
		
		for j in range(1, current_nodes + 1):
			# print("Delta(", l, ",", j, ") :", end = "")
			p_sum = 0
			for i in range(1, next_nodes + 1):
				p_sum += next_delta[i] * thetas[l][i][j]
			
			# element wise multiply of the theta' * x(i) .* a(l) * (1 - a(l))
			p_sum *= (activation_history[l][j] * (1 - activation_history[l][j]))
			
			current_delta.append(p_sum)
			# print(current_delta[-1])
		
		next_delta = current_delta
		next_delta.insert(0, 0)
		master_delta.append(next_delta)
		

		# print("Delta(", l, ") : ", master_delta[-1])
		# print("<================>")

	# removing the offset zeroes (optional)
	# master_delta = [ x[1:] for x in master_delta]
	
	# reversing and adding offset for labeling
	master_delta = [[], []] + master_delta[::-1]
	
	return master_delta

# this function accumilates the error across the training set
# returning the Capital Delta matrix 
def accumilate(CDel, activation_history, delta_history, m = 0):
	if m == 0:
		m = len(activation_history)
	
	# empty matrix
	CDelta = CDel
	theta_set = len(CDelta)
	
	for i in range(1, m + 1):
		cur_CDelta = CDel
		for l in range(1, theta_set):
			mat_build_row, mat_build_col = delta_history[i][l + 1], activation_history[i][l]
			cur_CDelta[l] = [ list(map(lambda x: x * i, mat_build_col)) for i in mat_build_row ]
		CDelta = helper.list_recur(CDelta, cur_CDelta)
		print(CDelta)
	return CDelta

# this regularizes the delta matrix and returns the partial derivatives of each param
def make_partial(m, thetas, CDeltas, lambd = 0):
	
	ad_term = helper.nest_operate(lambd, thetas, operator.mul)
	# in case regularization required
	if lambd:
		ad_term = [ [[j if i else 0 for i, j in enumerate(k)] for k in l] for l in ad_term[1:]] \
		# change the regular term according to theta_0 nomenclature
		ad_term.insert(0, []) # add offset for proper labeling
	
	CDeltas = helper.nest_operate((1 / m), CDeltas, operator.mul)

	# partial derivative matrix 
	D_Matrix = helper.list_recur(ad_term, CDeltas)
	return D_Matrix


def learn_theta(dataset, learning_rate, lambd, m, nodes_per, base_thetas, L):
	thetas = base_thetas
	theta_history = [ thetas ]
	CDelta_history = []	
	
	print("Inita Theta (θ) : ", thetas, sep = " : ")
	print("Press ENTER to continue ...")
	input()
	
	run_count = 1
	while True:
		print("RUN COUNT", run_count)
		print("* Learning Rate (α):", learning_rate) 
		master_history = [] # activation history run for the given theta
		master_delta = [] # error history for the given theta

		for t_ex in dataset:
			# forward propogate to get the activation values as well as final delta_L
			history, delta_L = FP(t_ex, thetas)
			master_history.append(history)
			# back propogate to get the delta matrix for all the examples and append to master_delta matrix
			delta_history = BP(delta_L, thetas, nodes_per, history, L)
			master_delta.append(delta_history)

		# offsetting the history to match labeling
		master_history.insert(0, [])
		master_delta.insert(0, [])
		
		# delta accumilator matrix
		CDelta = accumilate(make_theta(nodes_per, 0), master_history, master_delta, m)
		CDelta_history.append(CDelta)
		print("C_DEL AFTER RUN", CDelta, sep = "\n")
		
		# partial derivative matrix wrt induvidual theta
		partial_D = make_partial(m, thetas, CDelta, lambd)
		# print("PARTIAL", partial_D, sep = " : ")

		partial_D = helper.nest_operate(-learning_rate, partial_D, operator.mul)
		new_theta = helper.list_recur(thetas, partial_D, operator.sub)
		thetas = new_theta
		# print("Current theta (θ) : ", thetas, sep = " : ")
		theta_history.append(thetas)

		# break checks for various conditions
		
		# check for divergence
		if len(CDelta_history) > 1 and helper.check_divergence(CDelta_history[-1], CDelta_history[-2]):
			flag = 1
			break
		
		# check if thetas have converged
		if helper.close_enough(theta_history[-1], theta_history[-2]):
			flag = 0
			break
		
	
		run_count += 1

	if flag:
		print("THETAS HAVE DIVERGED")
		# escape function to be removed after dynamic learning rate adjustment
		return helper.nest_operate(0, base_thetas, operator.mul)
	else:
		print("\nThetas have been 'learnt' successfully\n")
		print("Final theta (θ) set : ", theta_history[-1], "\n")

	# finally return learnt thetas
	return thetas

def query_y(thetas):
	
	# this is for taking input from the user
	print("Enter the xi(s)", end = " : ")
	xi_s = list(map(float, list(filter( lambda x: x != '', input().strip(' ').split(' ')))))
	xi_s.insert(0,1)
	print("XI_S", xi_s)
	y_matrix = FP(xi_s, thetas, singleton = True)
	print(y_matrix)



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
	base_thetas = make_theta(nodes_per)
	
	final_theta = learn_theta(dataset, learning_rate, lambd, m, nodes_per, base_thetas, L)
	
	cont = True

	while cont:
		query_y(final_theta)
		print("Calculate another (Y/n) : ", end = "")
		ans, cont = str(input()), False
		if ans == "" or ans[0] == "" or ans[0] == "y":
			cont = True


if __name__ == '__main__':
	print("Neural Network dynamic implementation (v1.5)")
	main()
