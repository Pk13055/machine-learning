#!/home/pratik/anaconda3/bin/python3.6


'''
	This is an implementation of K-means algorithm to identify clustering
	(Uses numpy and matplotlib)
	Numpy is required, matplotlib is optional

'''

# system imports
from sys import argv as rd
import numpy as np
import config
import helper
try:
	import matplotlib.pyplot as plt
	is_graph = True
except:
	is_graph = False


# custom imports
import random
from math import sqrt
from statistics import mean
from time import sleep

# the function processes the dataset and chooses the initial centoids
def process_data(filename, K):
	raw_data = open(filename).read().strip(' ').split('\n')
	raw_data = helper.sanitize(raw_data)
	dataset = [[float(i) for i in x.strip(' ').split(' ')] \
		for x in raw_data]
	
	ind_range = range(len(dataset))
	indices = 2 * [ind_range]
	while len(indices) != len(set(indices)):
		indices = [random.choice(ind_range) for _ in range(K)]

	initial_centroids = [dataset[_] for _ in indices]

	return dataset, initial_centroids


def plotGraph(grouping, initial_c, final_c):
	n = len(final_c[0])
	
	if n < 2:
		return False

	else:
		x_ind, y_ind = (0, 0)
		r = lambda: random.randint(0,255)
		colors = []

		while x_ind == y_ind:
			x_ind, y_ind = random.choice(range(n)), random.choice(range(n))
		
		for dataset in grouping:
			color = "#%02X%02X%02X" % (r(),r(),r())
			colors.append(color)
			xi_s, yi_s = ([_[x_ind] for _ in dataset], [_[y_ind] for _ in dataset])
			plt.plot(xi_s, yi_s, color = color, marker = '.', linestyle = 'None')
		
		xi_center_old, yi_center_old = ([_[x_ind] for _ in initial_c], [_[y_ind] for _ in initial_c])

		xi_center_new, yi_center_new = ([_[x_ind] for _ in final_c], [_[y_ind] for _ in final_c])
		c_no = 1
		for i, j, k in zip(xi_center_new, yi_center_new, colors):
			plt.plot([i], [j], markerfacecolor = k, marker = 'D', linestyle = 'None', \
				markersize = '10' , markeredgecolor = 'black')
			plt.annotate(s = "Cluster %d" % (c_no), xy=(i, j), xytext=(-5, 10), \
				textcoords='offset points', ha='right', va='bottom',\
				bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
			c_no += 1

		plt.plot(xi_center_old, yi_center_old, 'rx')
		# plt.plot(xi_center_new, yi_center_new, 'gx')
		plt.title("x%d vs. x%d" % (y_ind + 1, x_ind + 1))
		plt.xlabel("x%d ->" % (x_ind + 1))
		plt.ylabel("x%d ->" % (y_ind + 1))

		plt.show()
		return True


# this functions applies K-means to find the centoids
def find_centroids(initial_centroids, dataset):

	K = len(initial_centroids)
	m = len(dataset)
	n = len(dataset[0])
	centroid_history = [initial_centroids] # this will record centroid convergence
	cluster_grouping = []
	
	run_count = 1
	while True:
		# stores the index of the centroid to which each example is close to
		close_to = []

		for ex in dataset:
			# measures the closeness of the current training example to each centroid
			# ith value is closeness to ith centroid
			closeness = []
			current_centroids = centroid_history[-1]
			for cluster in current_centroids:
				cur_close = sqrt(sum([(xi - ci) ** 2 for xi, ci in zip(ex, cluster)]))
				closeness.append(cur_close)
			# append the index of the closest centroid to the closeness history
			close_to.append(closeness.index(min(closeness)))

		print("#%d" % run_count, " :\t", close_to)
		sleep(1)
		
		# stores the grouped data
		cluster_grouping = [ [] for _ in range(K) ]

		# iterating and aggregating examples according to closeness
		[ cluster_grouping[clus_ind].append(dataset[x_index]) \
		for x_index, clus_ind in enumerate(close_to) ]

		# stores the new centroid values
		new_centroids = []
		# iterating over clusters of data
		# this loops creates the new centoids
		for clus in cluster_grouping:
			# current clus-th centroid
			current_centroid = [ mean([_[i] for _ in clus]) for i in range(n) ]
			new_centroids.append(current_centroid)
		centroid_history.append(new_centroids)
		run_count += 1

		# conditions to check convergence/divergence

		# break out if the centroids have converged
		if helper.is_close(centroid_history[-1], centroid_history[-2]):
			print("Centroids have converged")
			break

		# run overflow
		if run_count > config.max_run_count:
			print("Run Overflow")
			break

	# return the final centroids
	return centroid_history[-1], cluster_grouping


# tie everything together
def main():
	filename = rd[1]
	# number of clusters
	try:
		K = int(rd[2]) 
	except:
		K = config.default_K

	# convert raw file data into pythonic data
	dataset, initial_centroids = process_data(filename, K)
	# find centroid by applying K - means
	final_centroids, grouping = find_centroids(initial_centroids, dataset)
	if is_graph:
		plotGraph(grouping, initial_centroids, final_centroids)
	else:
		print("Final Centroids : \n", final_centroids)



# for script access
if __name__ == '__main__':
	print("K-means Implementation (v1.4)")
	main()