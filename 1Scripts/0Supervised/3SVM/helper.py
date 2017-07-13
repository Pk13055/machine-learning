'''

	The helper file contains all the helper functions which are not directly related to the script
	but play a part in it anyway. This is done so as to maintain a level of abstraction
	and keep the script clutter free.

'''
# default imports
import numpy as np
import config
import os

# custom imports for functions
import datetime

# sanitizes lists removing the 'char' elements
def sanitize(raw_group, char = ''):
	return type(raw_group)(filter(lambda x: x != char, raw_group))

# unpacks dicts into tuples
def unpack(dict_here):
	return tuple([dict_here[x] for x in dict_here])

# flatten the y list 
def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])