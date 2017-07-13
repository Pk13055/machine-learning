#!/home/pratik/anaconda3/bin/python3.6

'''
	This is a small python script to implement feature scaling, 
	mean normalization and then dimensionality reduction
	for k-means x cluster data

'''

# default imports
from sys import argv as rd
import numpy as np
from numpy import linalg

# custom imports
import os
import datetime
from time import sleep

# apply feature scaling and mean normalization
def process_data(filename, is_normal, ch = ' '):
	# this will hold the stats related to each row
	stats = []
	raw = open(filename).read().strip(' ').split('\n')
	raw = [x for x in raw if x != '']
	X = np.matrix([[float(_) for _ in x.strip(' ').split(ch)] for x in raw])
	m, n = X.shape
	if not is_normal:
		for i in range(n):
			x_i = X[:, i]
			stats.append((np.mean(x_i), np.std(x_i)))
			X[:, i] = (x_i - stats[-1][0]) / stats[-1][-1]
	return X, stats


# calculates K on the basis of the rentention
# set this higher for better accuracy, but lesser compression
# required (default is 95%) / (95 - 99%)
def calc_K(S, retention = 0.99):
	print("Calculating K for", retention * 100,"% retention of variance")
	sleep(1)
	n = S.shape[0]
	diag = np.ndarray.tolist(S) # S is already in diagonal form
	total_sum = sum(diag)
	for k in range(1, n):
		current_ret = sum(diag[:k]) / total_sum
		print("K :", k, "-> Rentention :", current_ret * 100)
		sleep(0.1)
		if current_ret >= retention:
			print("\nFinal dimensionality of Z matrix, K :", k)
			input("Press ENTER to continue")
			return k
	return n


# this function constructs a (m x k)
# repr of the data, ie compresses it
def Z_make(X, stats):
	m, n = X.shape

	sig = np.matrix([[0.0 for _ in range(n)] for _ in range(n)])
	for _ in range(m):
		x_i = X[_, :]
		sig += ((1 / m) * (x_i.T * x_i))
	
	U, S, V = linalg.svd(sig)
	k = calc_K(S)
	U_part = U[:, :k]

	return (U_part.T * X.T).T, U_part

# write the new Z data and reversal matrix to dataset
def write_file(Z, revert_matrix):
	cur_date = "_" + datetime.datetime.now().isoformat('_')[:-7] + '.txt'

	fo = open("Reversal_matrix" + cur_date, "a+")
	for _ in revert_matrix: # n x k
			for i in _:
				fo.write(str(i) + " ")
			fo.write("\n")
	fo.close()

	for _ in Z: # n x k
		fo = open("Z_matrix" + cur_date, "a+")
		for i in _:
			fo.write(str(i) + " ")
		fo.write("\n")
		fo.close()


def main():
	filename = rd[1]
	# get additional param for normalizing or not
	try:
		is_normal = rd[2]
	except:
		is_normal = False
	X, stats = process_data(filename, is_normal)
	Z, revert_mat = Z_make(X, stats)
	write_file(np.ndarray.tolist(Z), np.ndarray.tolist(revert_mat))


if __name__ == '__main__':
	print("PCA Dimensionality Reduction (v1.5)")
	main()

