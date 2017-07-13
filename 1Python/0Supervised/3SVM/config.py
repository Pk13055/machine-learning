'''

	This is the config file for the AIO script, it contains various
	constants and factors which you can use to alter the output of the script

'''

# n and m parameters to choose the best algorithm

# Linear kernel
linear_n = 1000
linear_m = 1000

# gaussian kernel
gauss_n = 1000
gauss_m = 10000

# add features
add_n = 1000
add_m = 10001

# record seperator
# each training example will be seperated by this char
r_sep = '\n'

# feature seperator
# features will be seperated by this char
f_sep = ' '

# default size of the y vector
def_y_vec_size = 1

# max size of multi-class classification
max_K = 15
# relative uniqueness 
relative_unique = 0.05

# split points for cross validation, training and test set
# in percentage
cross_ 	= 20
test_ 	= 20
train_ 	= 60

# recursion limit
recur_limit = 14900

# display params
max_display_X = 100
min_display_X = 10
max_example_print = 1000

# cache size for SVM implementation
cache_size = 1024

# C used for SVMs
C = 1
# Gamma (1 / 2 * sigma)
# default is auto
gamma = 'auto'