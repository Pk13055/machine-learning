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

- **0Octave**
 * [machine-learning-ex8](0Octave/machine-learning-ex8)
   * [ex8.pdf](0Octave/machine-learning-ex8/ex8.pdf)
   * [ex8](0Octave/machine-learning-ex8/ex8)
 * [machine-learning-ex7](0Octave/machine-learning-ex7)
   * [ex7.pdf](0Octave/machine-learning-ex7/ex7.pdf)
   * [ex7](0Octave/machine-learning-ex7/ex7)
 * [machine-learning-ex6](0Octave/machine-learning-ex6)
   * [ex6.pdf](0Octave/machine-learning-ex6/ex6.pdf)
   * [ex6](0Octave/machine-learning-ex6/ex6)
 * [machine-learning-ex5](0Octave/machine-learning-ex5)
   * [ex5.pdf](0Octave/machine-learning-ex5/ex5.pdf)
   * [ex5](0Octave/machine-learning-ex5/ex5)
 * [machine-learning-ex4](0Octave/machine-learning-ex4)
   * [ex4.pdf](0Octave/machine-learning-ex4/ex4.pdf)
   * [ex4](0Octave/machine-learning-ex4/ex4)
 * [machine-learning-ex3](0Octave/machine-learning-ex3)
   * [ex3.pdf](0Octave/machine-learning-ex3/ex3.pdf)
   * [ex3](0Octave/machine-learning-ex3/ex3)
 * [machine-learning-ex2](0Octave/machine-learning-ex2)
   * [ex2.pdf](0Octave/machine-learning-ex2/ex2.pdf)
   * [ex2](0Octave/machine-learning-ex2/ex2)
 * [machine-learning-ex1](0Octave/machine-learning-ex1)
   * [ex1.pdf](0Octave/machine-learning-ex1/ex1.pdf)
   * [ex1](0Octave/machine-learning-ex1/ex1)

- **1Scripts**
 * [0Supervised](1Scripts/0Supervised)
   * [0Linear](1Scripts/0Supervised/0Linear)
     * [dataset_gen.py](1Scripts/0Supervised/0Linear/dataset_gen.py)
     * [grad-descent.py](1Scripts/0Supervised/0Linear/grad-descent.py)
     * [normalization.py](1Scripts/0Supervised/0Linear/normalization.py)
     * [RUN.md](1Scripts/0Supervised/0Linear/RUN.md)
   * [3SVM](1Scripts/0Supervised/3SVM)
     * [np_SVM.py](1Scripts/0Supervised/3SVM/np_SVM.py)
     * [RUN.md](1Scripts/0Supervised/3SVM/RUN.md)
   * [1Neural](1Scripts/0Supervised/1Neural)
     * [intial.py](1Scripts/0Supervised/1Neural/intial.py)
     * [np_initial.py](1Scripts/0Supervised/1Neural/np_initial.py)
     * [RUN.md](1Scripts/0Supervised/1Neural/RUN.md)
   * [Structure.md](1Scripts/0Supervised/Structure.md)
 * [1Unsupervised](1Scripts/1Unsupervised)
     * [dataset_gen.py](1Scripts/1Unsupervised/dataset_gen.py)
     * [PCA.py](1Scripts/1Unsupervised/PCA.py)
     * [0K-Means](1Scripts/1Unsupervised/0K-Means)
       * [anomaly_detect.py](1Scripts/1Unsupervised/0K-Means/anomaly_detect.py)
       * [k_means.py](1Scripts/1Unsupervised/0K-Means/k_means.py)
       * [image_compress.py](1Scripts/1Unsupervised/0K-Means/image_compress.py)
       * [RUN.md](1Scripts/1Unsupervised/0K-Means/RUN.md)
     * [1Recomender-Systems](1Scripts/1Unsupervised/1Recomender-Systems)
       * [dataset_gen.py](1Scripts/1Unsupervised/1Recomender-Systems/dataset_gen.py)
       * [recommend.py](1Scripts/1Unsupervised/1Recomender-Systems/recommend.py)
       * [RUN.md](1Scripts/1Unsupervised/1Recomender-Systems/RUN.md)
     * [RUN.md](1Scripts/1Unsupervised/RUN.md)
