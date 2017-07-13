# Machine Learning Scripts

## Introduction

This is a collection of scripts (implemented in Octave and python (_vanilla, numpy and/or matplotlib_)) 
Everything is well documented, and has sample test data to try out the scrips with. The scripts range from supervised to unsupervised, covering everything from simple gradient descent to more complex neural network implemetations.
Furthermore, some scripts are written in both numpy as well as Vanilla python to support cross compatibility.

## How to run

- Every directory contains a relevant `*.md` file that contains the instructions to run the scripts in that file. 
- Most scripts come with a `dataset generation script`, that will help you to generate a _random_ dataset for the specific algortihm.
- In addition to this, a few predefined datasets have also been included.
- All code is well documented and commented for easy readability and understanding.
- Linux preferred with python 3.x support. (Numpy and matplotlib are **essential** for a few scripts)

## Octave Exercises

- This folder contains all the assignments and exercises covered by the _Machine learning course_ by **Andrew Ng** on Coursera. They have pfs as well as the solutions for every exercise.
- You can use Octave **OR** Matlab to implement the exercises given.

## Directory Structure 

1Python
 * [__init__.py](1Python/__init__.py)
 * [0Supervised](1Python/0Supervised)
   * [0Linear](1Python/0Supervised/0Linear)
     * [config.py](1Python/0Supervised/0Linear/config.py)
     * [data_big_normal.txt](1Python/0Supervised/0Linear/data_big_normal.txt)
     * [data_big.txt](1Python/0Supervised/0Linear/data_big.txt)
     * [data_class_normal.txt](1Python/0Supervised/0Linear/data_class_normal.txt)
     * [data_class.txt](1Python/0Supervised/0Linear/data_class.txt)
     * [data_reg_normal.txt](1Python/0Supervised/0Linear/data_reg_normal.txt)
     * [data_reg.txt](1Python/0Supervised/0Linear/data_reg.txt)
     * [dataset_gen.py](1Python/0Supervised/0Linear/dataset_gen.py)
     * [grad-descent.py](1Python/0Supervised/0Linear/grad-descent.py)
     * [normalization.py](1Python/0Supervised/0Linear/normalization.py)
     * [RUN.md](1Python/0Supervised/0Linear/RUN.md)
     * [__init.py__](1Python/0Supervised/0Linear/__init.py__)
     * [__init__.py](1Python/0Supervised/0Linear/__init__.py)
   * [Datasets](1Python/0Supervised/Datasets)
   * [3SVM](1Python/0Supervised/3SVM)
     * [helper.py](1Python/0Supervised/3SVM/helper.py)
     * [Actual.txt](1Python/0Supervised/3SVM/Actual.txt)
     * [config.py](1Python/0Supervised/3SVM/config.py)
     * [dataset.txt](1Python/0Supervised/3SVM/dataset.txt)
     * [image_data.txt](1Python/0Supervised/3SVM/image_data.txt)
     * [np_SVM.py](1Python/0Supervised/3SVM/np_SVM.py)
   * [1Neural](1Python/0Supervised/1Neural)
	   * [__init__.py](1Python/0Supervised/1Neural/__init__.py)
	   * [RUN.md](1Python/0Supervised/1Neural/RUN.md)
	   * [backup.py](1Python/0Supervised/1Neural/backup.py)
	   * [intial.py](1Python/0Supervised/1Neural/intial.py)
	   * [helper.py](1Python/0Supervised/1Neural/helper.py)
	   * [np_helper.py](1Python/0Supervised/1Neural/np_helper.py)
	   * [image_data.txt](1Python/0Supervised/1Neural/image_data.txt)
	   * [dataset.txt](1Python/0Supervised/1Neural/dataset.txt)
	   * [np_initial.py](1Python/0Supervised/1Neural/np_initial.py)
	   * [config.py](1Python/0Supervised/1Neural/config.py)
	   * [normalized_data.txt](1Python/0Supervised/1Neural/normalized_data.txt)
	   * [reduced_image_data.txt](1Python/0Supervised/1Neural/reduced_image_data.txt)
 * [1Unsupervised](1Python/1Unsupervised)
     * [dataset_gen.py](1Python/1Unsupervised/dataset_gen.py)
     * [image_data.txt](1Python/1Unsupervised/image_data.txt)
     * [dataset.txt](1Python/1Unsupervised/dataset.txt)
     * [image_data_Y.txt](1Python/1Unsupervised/image_data_Y.txt)
     * [image_data_X.txt](1Python/1Unsupervised/image_data_X.txt)
     * [Reversal_matrix_2017-07-13_08:07:26.774323.txt](1Python/1Unsupervised/Reversal_matrix_2017-07-13_08:07:26.774323.txt)
     * [Z_matrix_2017-07-13_08:07:26.774323.txt](1Python/1Unsupervised/Z_matrix_2017-07-13_08:07:26.774323.txt)
     * [reduced_image_data.txt](1Python/1Unsupervised/reduced_image_data.txt)
     * [PCA.py](1Python/1Unsupervised/PCA.py)
     * [Datasets](1Python/1Unsupervised/Datasets)
     * [0K-Means](1Python/1Unsupervised/0K-Means)
         * [__init__.py](1Python/1Unsupervised/0K-Means/__init__.py)
         * [helper.py](1Python/1Unsupervised/0K-Means/helper.py)
         * [config.py](1Python/1Unsupervised/0K-Means/config.py)
         * [dataset.txt](1Python/1Unsupervised/0K-Means/dataset.txt)
         * [RUN.md](1Python/1Unsupervised/0K-Means/RUN.md)
         * [k_means.py](1Python/1Unsupervised/0K-Means/k_means.py)