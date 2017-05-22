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

# calculates the regression params for the given run
theta = [randint(- 10 ** 6, 10 ** 6) for i in range(n + 1)]

# to calculate yi on the basis of the thetas and xi
def h(xi_s):
	return sum([i * j for i, j in zip(xi_s, theta)])

# ties everything together
def main():
	if os.path.exists(file_name):
		os.remove(file_name)
	for i in range(m):
		fout = open(file_name, "a+")
		xi_s = [randint(10 ** 3, 10 ** 9) for i in range(n)]
		xi_s.insert(0, 1)
		for _ in xi_s:
			fout.write(str(_) + " ")
		fout.write(str(h(xi_s)) + " \n")
		fout.close()

# enables script access
if __name__ == '__main__':
	main()