#!/home/pratik/anaconda3/bin/python3.6

''' 
	this is a dataset generator that generates training examples 
	for unsupervised learning algorithms.

'''

from sys import argv as rd
import os
import random
from operator import add, sub
try:
	import matplotlib.pyplot as plt
	is_plot = True
except:
	is_plot = False 


# this is the factor range in which all the features will lie
# change this to scale factors
factor_scale = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]

# this function takes in the clusters and the feature size and 
# creates a not-so-random random cluster set group
def create_data(m, n, K):
	training_data = []

	# scaling of random factors
	current_scale = [random.choice(factor_scale) for _ in range(n)]

	# the centroid list for K clusters
	cluster_centroids = [[random.random() * current_scale[_] for _ in range(n)] for k in range(K)]
	
	for _ in range(m):
		closest_cluster = random.choice(cluster_centroids)
		current_ex = [random.choice([add, sub])(closest_cluster[i], \
			random.random() * current_scale[i]) for i in range(n)]
		training_data.append(current_ex)
	
	return training_data, cluster_centroids

def plotData(raw_data, centroids):
	n = len(centroids[0])
	if n < 2:
		return False

	else:
		x_ind, y_ind = random.choice(range(n)), random.choice(range(n))
		if x_ind == y_ind:
			y_ind = (y_ind + 1) % n
		xi_s, yi_s = ([_[x_ind] for _ in raw_data], [_[y_ind] for _ in raw_data])
		xi_center, yi_center = ([_[x_ind] for _ in centroids], [_[y_ind] for _ in centroids])

		plt.plot(xi_s, yi_s, 'b.')
		plt.plot(xi_center, yi_center, 'rx')
		plt.title("x%d vs. x%d" % (y_ind + 1, x_ind + 1))
		plt.xlabel("x%d ->" % (x_ind + 1))
		plt.ylabel("x%d ->" % (y_ind + 1))

		plt.show()
		return True


# getting the inputs and writing to file
def main():
	filename = rd[1]
	n = int(rd[2]) # feature size of every example
	m = int(rd[3]) # number of training examples
	# number of clusters
	try:
		K = int(rd[4])
	except:
		K = 2
	
	if K > m:
		print("Number of clusters cannot be greater than number of \
			training examples. Exiting")
		return False

	raw_data, centroids = create_data(m, n, K) # pythonic dataset
	
	# delete the file if it exists
	if os.path.exists(filename):
		os.remove(filename)
	
	# rapid input output might be slower, but will
	# prevent buffer overflow, enabling large feature sets
	for train_ex in raw_data:
		fo = open(filename, 'a+')
		for x_i in train_ex:
			fo.write(str(x_i) + " ")
		fo.write("\n")
		fo.close()
	
	# plot the data if matplot can be imported
	if is_plot:
		plotData(raw_data, centroids)
	else:
		print("\nArtificial Centroids : ", centroids)



# for script access
if __name__ == '__main__':
	main()