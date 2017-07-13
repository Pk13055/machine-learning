#!/home/pratik/anaconda3/bin/python3.6

'''
	this script takes in an image and outputs a compressed K-bit image
	(uses the K-means algorithm)

'''
# system imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sys import argv as rd
from PIL import Image
from pathlib2 import Path
import k_means

# custom imports
from time import sleep

SUPPORTED_IMAGES = set(['png', 'jpg', 'jpeg'])

# this reconstruct the RGB values to have only
# centroids rows (param R/G/B)
# this should be in lofl form
def reconstruct(mat, K):
	mat_centroids = k_means.find_centroids(mat, K)
	
	# takes a lofl and returns a lofl
	return [mat_centroids[i] for i in \
		[k_means.closest_to(_, mat_centroids) for _ in mat ]]

# function that compresses the data
def compress(input_file, K_list, output_file):
	img = mpimg.imread(input_file)
	K = [K_list]	
	
	# getting RGB values
	
	print("Reconstructing R values ...")
	R = np.ndarray.tolist(img[:, :, 0])
	R_reconstructed = np.array(reconstruct(R, K))
	
	print("Reconstructing G values ...")
	G = np.ndarray.tolist(img[:, :, 1])
	G_reconstructed = np.array(reconstruct(G, K))
	
	print("Reconstructing B values ...")
	B = np.ndarray.tolist(img[:, :, 2])
	B_reconstructed = np.array(reconstruct(B, K))

	print("Trying to Reconstruct A values")
	try:
		A = np.ndarray.tolist(img[:, :, 3])
		A_reconstructed = np.array(reconstruct(A, K))
		is_A = True
	except:
		print("No alpha values")
		is_A = False

	print("Reconstructing Image")
	img_array = [R_reconstructed, G_reconstructed, B_reconstructed]
	
	if is_A:
		img_array.append(A_reconstructed)

	img_reconstructed = np.dstack(img_array)
	img_reconstructed.dtype = 'float64'
	print("Image Dimensions : ", img_reconstructed.shape)
	input("PRESS ENTER TO CONTINUE")
	
	plt.imshow(img_reconstructed)
	plt.show()


# main function to get input file etc
def main():
	input_file = rd[1]
	compression = int(rd[2])
	
	# exit out if file not supported
	if input_file.rsplit('.', 1)[1].lower() not in SUPPORTED_IMAGES:
		print("Input file %s not supported" % (input_file))
		return False
	
	output_file = input_file.rsplit(',')[0] + "_compressed_%d_bit" %\
	 compression + Path(input_file).suffix
	compress(input_file, compression, output_file)


# for script access
if __name__ == '__main__':
	print("Image Compression (v1.0)")
	main()