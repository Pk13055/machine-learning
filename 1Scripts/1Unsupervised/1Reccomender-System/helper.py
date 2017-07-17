'''
	Helper functions to be used in the recommender system
	but not directly related to the algorithm

'''
import config

# this function removes any oddities in the data set
def sanitize(unset_data):
	return type(unset_data)(filter(lambda x: x != '', unset_data))

# the function processes the dataset
def process_data(filename, ch = ' '):
	raw_data = open(filename).read().strip(' ').split('\n')
	raw_data = sanitize(raw_data)
	return [[float(i) for i in x.strip(' ').split(ch)] \
		for x in raw_data]


# check closeness of J values
def close_enough(new_J, old_J):
	return abs(new_J - old_J) / new_J < config.J_tolerance