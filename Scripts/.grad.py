#!/usr/bin/python3
import sys, random
from random import randint

t0 = 0
t1 = 0
m = int(sys.argv[1])
# house_list = [[randint(100, 10 ** 5), randint(10 ** 5, 10 ** 9)] for _ in range(m)]
house_list = [ [i, 2 * i] for i in range(10 * m, 100 * m)]
def h(size, t0, t1):
	return t0 + size * t1

learning_rate = 0.0001
PRE = 0.01
try:
	learning_rate = int(sys.argv[2])
	PRE = int(sys.argv[3])
except:
	pass

def calc(t0 = t0, t1 = t1):
	while True:
		temp_sum = [0, 0]
		for _ in range(m):
			temp_sum[0] += ( h(house_list[_][0], t0, t1) - house_list[_][1])
			temp_sum[1] += (( h(house_list[_][0], t0, t1) - house_list[_][1]) * house_list[_][0])
		temp0 = t0 - learning_rate / m * temp_sum[0]
		temp1 = t1 - learning_rate / m * temp_sum[1]
		print(temp0, temp1)
		if abs(temp0 - t0) < PRE and abs(temp1 - t1) < PRE:
			break
		else:
			t0 = temp0
			t1 = temp1
	return temp0, temp1

def main():
	t0, t1 = calc()
	print(t0, t1)
	for _ in range(m):
		print("House size", house_list[_][0], "cost", house_list[_][1], sep = " : " )
	for _ in range(int(input())):
		print("Enter the size", end = " : ")
		size = int(input())
		print("The predicted cost is : " + str(h(size, t0, t1)))

if __name__ == '__main__':
	main()



