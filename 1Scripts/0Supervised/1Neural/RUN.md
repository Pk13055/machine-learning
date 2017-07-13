# Neural Network Implementation 
Neural network has been implementeed using basic python. It is an implementation of the backpropogation algorithm
This script has the following features:
	- Support for normalized data for faster convergence
	- Multi-layered node and layers
	- K - dimensional output vector support

## How to run
- ` ./intial.py <hidden layers> <dataset_file> <k*> <normalized_data*> `
- The breakdown:
	- hidden layers: The number of hidden layers you wish to include in your neural network
	- k: The dimension of the output vector of your dataset. (Param depends on the dataset) (DEFAULT is one)
	- normalized_data: The normalized file for your dataset