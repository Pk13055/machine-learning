#!/home/pratik/anaconda3/bin/python3.6

'''

	This script is designed to detect anomalies in a given dataset
	Python 3.x recommended along with numpy and matplotlib modules

'''

# default imports
from sys import argv as rd
import numpy as np
import helper
try:
	import matplotlib.pyplot as plt
	is_graph = True
except:
	is_graph = False

# custom imports
from math import pi, exp, sqrt
from time import sleep

# the function processes the dataset
def process_data(filename, ch = ' '):
	raw_data = open(filename).read().strip(' ').split('\n')
	raw_data = helper.sanitize(raw_data)
	return [[float(i) for i in x.strip(' ').split(ch)] \
		for x in raw_data]

# main function that calculates the means and devs
# returns dev_mat (s1, s2, s3 ... sn), mean_mat (u1, u2 ... un)
def get_stats(dataset):
	m, n = dataset.shape
	dev_mat = np.matrix([np.std(dataset[:, _]) for _ in range(n)])
	mean_mat =  np.matrix([np.mean(dataset[:,_]) for _ in range(n)])
	return mean_mat, dev_mat

# calculates the gaussian probability for a given x, u, s
def gauss_prob(xi, u, sigma):
	return exp(- (xi - u) ** 2 / (2 * (sigma  ** 2))) / (sqrt(2 * pi) * sigma)
gauss_prob = np.vectorize(gauss_prob)


# flatten one deep nested lofl ONLY 
def flatten(S):
	for _ in S:
		for y in _:
			yield y



# visualize the data and calculates stats
def visualize(all_p, means, devs, tolerance):
	count = 1
	total_faults = 0
	fault_array = [[] for _ in range(all_p.shape[-1])]
	m, n = all_p.shape
	max_len = 0
	for _ in all_p:
		prob1 = gauss_prob(_, means, devs)
		vals = np.ndarray.tolist(prob1 > tolerance)[0]
		faults = [i for i in range(len(vals)) if vals[i]]
		if len(faults):
			print("Anomalie(s) for x%d in features :" % count, faults)
			if len(faults) >= max_len:
				indx = (count, faults)
				max_len = len(faults)
			[fault_array[x].append(count) for x in faults]
			total_faults += 1
			if not ("-f" in rd or "--fast" in rd):
				sleep(0.1)
		count += 1
	
	if not total_faults:
		print("No Anomalies!")
	else:
		print("PERCENTAGE FAULTS", (total_faults / (all_p.shape[0] * all_p.shape[1])) * 100)
		# fault array contains the indices of all the xi for each given fault
		flat_fault = list(flatten(fault_array))
		
		fault_percent = {}
		for i in set(flat_fault):
			fault_percent["x(" + str(i) + ")"] = (flat_fault.count(i) / total_faults)
		
		if is_graph: 
			
			plt.subplot(2,2,1)
			plt.title("Anomaly distribution by example number")
			plt.axis('equal')
			if len(fault_percent) < 0.03 * m:
				labels = [_ for _ in fault_percent]
			else:
				labels = None
			fault_percent_list = [fault_percent[_] for _ in fault_percent]
			plt.pie(fault_percent_list, \
				labels = labels)

			plt.subplot(2,2,2)
			plt.title("Anomaly distribution by feature numbers")
			max_h = max([len(x) for x in fault_array])
			plt.axis([0, n, 0, 1.05 * max_h])
			plt.ylabel("Number of Anomalies ->")
			plt.xlabel("Features ->")
			plt.bar(left = [x for x in range(len(fault_array))], height = [len(x) for x in fault_array])

			plt.subplot(2,2,3)
			x, feature_fault = indx
			plt.title("Most Anomalous example breakdown x(%d)" % x)
			plt.axis([0, n, 0, 1])
			plt.xlabel("Features ->")
			heights = [1 for _ in range(len(feature_fault))]
			plt.bar(left = feature_fault, height = heights)
			
			plt.subplot(2,2,4)
			indx2 = [len(x) for x in fault_array].index(max_h)
			plt.title("Most Anomalous feature breakdown, f(%d)" % indx2)
			faulters = fault_array[indx2]
			sample = [1 for _ in range(len(faulters))]
			if len(faulters) > 0.03 * m:
				faulters = None
			plt.axis('equal')
			plt.pie(sample, labels = faulters)

			plt.tight_layout(w_pad = 2)
			plt.show()

		else:
			print(fault_percent, fault_array, sep = "\n")


# 0.6782 threshold for large dataset
# once calculated check anomalies
def query_y(means, devs, tolerance):

	# this is for taking input from the user
	print("Enter the xi(s)", end = " : ")
	try:
		xi_s = np.matrix(list(map(float, \
			list(filter( lambda x: x != '', \
				input().strip(' ').split(' '))))))
		print("I/P (x) -> ", xi_s)
		if xi_s.shape != means.shape:
			raise Exception
		
		m, n = xi_s.shape
		normal_prob = gauss_prob(xi_s, means, devs)
		print("P(x; u, std) ->",  normal_prob)
		[print("Anomaly in x%d " % (x + 1)) for x, j \
			in enumerate(np.ndarray.tolist(normal_prob \
				> tolerance)[0]) if j]
		return True
	
	except:
		print("Invalid Input")
		return False 


# tie everything together
def main():
	filename = rd[1]
	# get the tolerance for anomaly config
	# higher is more "forgiving"
	try:
		tolerance = float(rd[2])
	except:
		tolerance = 0.3
	# whether csv or space separated
	sep_char = " "
	
	dataset = np.matrix(process_data(filename, sep_char))
	mean_mat, devs_mat = get_stats(dataset)
	if not ("--no-graph" in rd):
		visualize(dataset, mean_mat, devs_mat, tolerance)

	query_loop = True
	while query_loop:
		query_y(mean_mat, devs_mat, tolerance)		
		print("Calculate another (Y/n) : ", end = "")
		ans, query_loop = str(input()), False
		if ans == "" or ans[0] == "" or ans[0] == "y":
			query_loop = True


# script access
if __name__ == '__main__':
	print("Anomaly Detection (v1.9)")
	main()