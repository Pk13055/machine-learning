#!/home/pratik/anaconda3/bin/python3.6
"""
	Generate random user database
	n users valuating m things
	resulting in a m x n dataset
	(space-separated)

"""
from sys import argv as rd
from random import choice, random
from os import remove
from os.path import exists

def r(x):
	return round(x * 2.0) / 2.0

# n users evaluating m things
def createData(n, m, fillage):
	dataset = [[0 for _ in range(n)] for _ in range(m)]
	choices_to_make = int(fillage * n * m)
	for _ in range(choices_to_make):
		x, y = (choice(range(m)), choice(range(n)))
		while dataset[x][y]:
			x, y = (choice(range(m)), choice(range(n)))
		dataset[x][y] = r(10 * random())
	return dataset

# write data to the file
def write_file(filename, dataset):
	m, n = len(dataset), len(dataset[0])
	if exists(filename):
		remove(filename)
	for row in dataset:
		fo = open(filename, 'a+')
		for col in row:
			fo.write(str(col) + " " )
		fo.write("\n")
		fo.close()
			

def main():
	filename = rd[1]
	n = int(rd[2])
	m = int(rd[3])
	# how much of the data set should be full
	try:
		fillage = int(rd[4])
	except:
		fillage = 0.4
	write_file(filename, createData(n, m, fillage))


if __name__ == '__main__':
	main()