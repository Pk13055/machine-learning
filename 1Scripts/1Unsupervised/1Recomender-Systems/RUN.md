# Recommender Systems

## Introduction 

The script _recommend.py_ is an implementation of the recommender algorithm. Based on an input dataset, the algorithm learns features for each training examples, and automatically fills in the _blank_ values. It, then, is able to "predict" what certain users may or may not like.

## How to run 

- `./recommend.py filename n learning_rate 	regularization param`
- The breakdown:
	- _filename_: The filename of the input dataset.
	- _n_: The feature size for each of training example. (_larger is more computationally expensive_)
	- _learning rate_: The learning rate of the algorithm. (_larger learning rates may diverge_)
	- _regularization param_: The regularization param. (_default is 0.2_)

## Project Tree
.
 * [dataset_gen.py](./dataset_gen.py)
 * [recommend.py](./recommend.py)
 * [RUN.md](./RUN.md)