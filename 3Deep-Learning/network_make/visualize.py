#!/home/pratik/anaconda3/bin/python3.6

''' 
	Visualzing the data
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sys import argv as rd
import pickle
import random
from data_lib import * 



def draw_person(person, pose_no):
	plt.clf()
	fig = plt.figure()
	ax = fig.gca(projection = '3d')
	for _ in range(person.shape[0]):
		ax.scatter(person[_, 0], person[_, 2],person[_, 1])
		ax.text(person[_, 0], person[_, 2], person[_, 1], point_labels[_],\
		 size = 12, zorder = 1)
	ax.set_title(pose_no)
	ax.set_ylim(1, 3)
	plt.show()

def get_person(ind):
	return X[ind, :-2].reshape(25,3)

def get_pose_indices(pose):
	return random.choice([ k for k, j in enumerate(y) if j[0,0] == pose])

def main():
	# pose = int(rd[1])
	for pose in range(1, 12 + 1):
		_ = get_pose_indices(pose)
		draw_person(get_person(_), "POSE " + str(pose))
		plt.show()

if __name__ == '__main__':
	main()