'''
	This file provides the various (non-cli) parameters for the gradient descent script to work
	Values have already been filled in with the default values, change as necessary
'''
## DEFAULT FILENAMES

# default normalized data file name 
default_normal_filename = "normal_data.txt"

## TOLERANCES

# this is the tolerance to check for the divergence of theta, set to suitable
# value (lower the better)
tolerance = 0.00001

# tolerance for the J difference
J_tolerance = 0.001

# dynamic increase alpha to speed up regression
dyn_alpha_tolerance = 0.001

## MISC

# regularization parameter
# default set to 0, ie unregularized
regularization_param = 0

# the timeout which controls the overall time (in seconds) for each run 
# for a specific value of theta
timeout = 30

# time to pause between change of theta
pause_theta = 1

# the J value which decides that the cost cannot be minimized
min_J_cost = 1e-10

## ITERATION CONTROLLERS

# minimum number of different thetas to try
min_run_count = 5

# maximum number of thetas to try
max_run_count = 500

# max number of iterations per theta (higher the better, may be slow for larger datasets)
max_iterations = 20000

## Multiplication Factors

# Divergence factor to divide by (when cost increases)
J_divergence_factor = 10

# factor to multiply by incase of large number of iterations (larger can lead to convergence)
iter_overflow_factor = (10 / 3)

# factor to multiply incase of overflow
time_exceed_factor = 3

# factor to multiply by in case of betterment of theta
better_theta_factor = 3

# factor to increase alpha dynamically 
dyn_alpha_factor = 1.05

## CLASSIFICATION PARAMETERS ##

# max cost for classification 
max_cost = 10 ** 9
# maxmium thetas for a classification problem
class_run_count = 20
# iteration count for classes
class_iter_count = 10000