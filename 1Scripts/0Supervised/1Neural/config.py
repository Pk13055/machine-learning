''' 
	This is the configuration file that contains the defaults for the neural network implementation.
	Change values here with caution.

'''
# learning rate alpha
learning_rate = 1e-5
k = 1

# regularization parameter
lambd = 0

# normal file to search for
normal_file = "normal_text.txt"

# tolerance
tolerance = 1e-8


theta_tolerance = 1e-8
J_tolerance = 1e-6
J_max = 1e30
J_break = 1e-9
# epsilon value
epsilon = 1e-4


max_run_count = 10000
min_run_count = 1000