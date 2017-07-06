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


# custom imports


def main():
	filename = rd[1]
	# number of clusters
	try:
		K = int(rd[2]) 
	except:
		K = config.default_K



# for script access
if __name__ == '__main__':
	print("K-means Implementation (v1.0)")
	main()