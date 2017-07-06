# helper for np specific implementation
import numpy as np

def close_theta(new_theta, old_theta, tolerance):
	error_mat = [np.all(((abs(x - y) / abs(x)) * 100) < tolerance) \
		for x, y in zip(new_theta[1:], old_theta[1:])]
	return all(error_mat)