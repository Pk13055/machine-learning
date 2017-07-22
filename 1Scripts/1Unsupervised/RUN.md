# Unsupervised Learning

## Introduction

This directory contains most scripts associated with unsupervised learning. General scripts that work with algorithms
across different categories have been put into this folder, with other specific algorithms in their own dirs.

## Scripts and Folders

- _0K-Means_: This folder contains the scripts associated with the **k-means clustering algorithm**.
- _1Recommender-Systems_: This folder contains the **recommender system** algorithm, used widely today to study
user trends and preferences.
- *dataset_gen.py*: This script is used to generate _normalized*_, _not-so-random_ random datasets to be used across the board with unsupervised learning scripts.
- _PCA.py_: This script is the implementation of the **Principle Component Analysis** algorithm. It is used to reduce _m_ dimensional data into _k_ dimensions.

## How to run

1. `./dataset_gen.py filename n m K`
	- _filename_: The name of the dataset to which the dataset is to be saved.
	- _n_: Number of features per training example.
	- _m_: Number of training examples.
	- _K*_: Number of clusters (_optional_)

2. `./PCA.py filename is_normal`
	- _filename_: The filename of input dataset.
	- *is_normal*: Whether or not the data is already normalized. (_default: False_)

## Project tree

.
 * [dataset_gen.py](./dataset_gen.py)
 * [PCA.py](./PCA.py)
 * [0K-Means](./0K-Means)
   * [anomaly_detect.py](./0K-Means/anomaly_detect.py)
   * [k_means.py](./0K-Means/k_means.py)
   * [image_compress.py](./0K-Means/image_compress.py)
   * [RUN.md](./0K-Means/RUN.md)
 * [1Recomender-Systems](./1Recomender-Systems)
   * [dataset_gen.py](./1Recomender-Systems/dataset_gen.py)
   * [recommend.py](./1Recomender-Systems/recommend.py)
   * [RUN.md](./1Recomender-Systems/RUN.md)
 * [RUN.md](./RUN.md)