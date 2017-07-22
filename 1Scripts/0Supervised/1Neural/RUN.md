# Neural Network Implementation 
Neural network has been implementeed using basic python. It is an implementation of the backpropogation algorithm
This script has the following features:
	- Support for normalized data for faster convergence
	- Multi-layered node and layers
	- K - dimensional output vector support
However, this implementation has not been tested and debugged properly, use at your own risk.

In addition to the _Vanilla_ implementation, another (_faster_) implementation has been done in `numpy` under the
`np_inital.py` script name. 

## How to run

### Vanilla Implementation

- ` ./intial.py <hidden layers> <dataset_file> <k*> <normalized_data*> `
- The breakdown:
	- hidden layers: The number of hidden layers you wish to include in your neural network
	- k: The dimension of the output vector of your dataset. (Param depends on the dataset) (DEFAULT is one)
	- normalized_data: The normalized file for your dataset

### `Numpy` Implementation

- `./np_inital.py learning_rate hidden_layers filename <regularization param> <output nodes>`
- The breakdown:
	- _learning rate_: The rate at which your algorithm learns (_== alpha_)
	- _hidden layers_: The number of hidden layers you wish to include in your neural network
	- _filename_ : The filepath of your dataset
	- _regularization param_: Lambda for regularization (_if left blank, default 0 used_)
	- _output nodes_: The length of your output _y_ vector. (_default is **1**_)

## Project Tree
.
 * [__init__.py](./__init__.py)
 * [intial.py](./intial.py)
 * [helper.py](./helper.py)
 * [np_helper.py](./np_helper.py)
 * [config.py](./config.py)
 * [np_initial.py](./np_initial.py)
 * [RUN.md](./RUN.md)