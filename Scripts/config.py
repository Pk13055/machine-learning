'''
	This file provides the various (non-cli) parameters for the gradient descent script to work
	Values have already been filled in with the default values, change as necessary
'''

# default normalized data file name 
default_normal_filename = "normal_data.txt"

# this is the tolerance to check for the divergence of theta, set to suitable
# value (lower the better)
tolerance = 0.0000001

# the timeout which controls the overall time (in seconds) for each run 
# for a specific value of theta
timeout = 30

# minimum number of different thetas to try
min_run_count = 5
# maximum number of thetas to try
max_run_count = 1000

# max number of iterations per theta (higher the better, may be slow for larger datasets)
max_iterations = 20000

# DIvergence factor to divide by (when cost increases)
J_divergence_factor = 10
# factor to multiply by incase of large number of iterations (larger can lead to convergence)
iter_overflow_factor = (10 / 3)

# time to pause between change of theta
pause_theta = 1
