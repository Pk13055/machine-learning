#!/usr/bin/python3

'''

	This generator generates not-so-random random data sets
	Uses a liner regression equation to get exact data points, albeit random-valued

'''

import datetime, sys, os
from random import randint

file_name = sys.argv[1]
n = int(sys.argv[2])
m = int(sys.argv[3])
is_class = "n"

try:
	is_class = sys.argv[4]
	from random import choice, random
	from math import e, ceil
except:
	is_class = "n"


if is_class in ["n", "no", "nil", ""]:
	is_class = False

# calculates the regression params for the given run
if is_class:
	theta = [choice([-10, 10]) * random() for i in range(n + 1)]
	print(str(theta))
else:
	theta = [randint(- 10 ** 6, 10 ** 6) for i in range(n + 1)]

# sigmoid function
def sigmoid(h_value):
	return 1 / (1 + e ** -h_value)

# to calculate yi on the basis of the thetas and xi
def h(xi_s):
	h_value = sum([i * j for i, j in zip(xi_s, theta)])
	print(h_value)
	if is_class:
		return round(sigmoid(h_value))
	else:
		return h_value
		

# ties everything together
def main():
	if os.path.exists(file_name):
		os.remove(file_name)
	for i in range(m):
		fout = open(file_name, "a+")
		if is_class:
			xi_s = [random() for _ in range(n)]
		else:
			xi_s = [randint(10 ** 3, 10 ** 9) for i in range(n)]
		xi_s.insert(0, 1)
		for _, i in enumerate(xi_s):
			if is_class and i != 0:
				_ = choice([-10, 10]) * _
			fout.write(str(_) + " ")
		fout.write(str(h(xi_s)) + " \n")
		fout.close()

# enables script access
if __name__ == '__main__':
	main()