# Machine Learning Journey

This repo is all the stuff (extra) that I've done on my journey in machine learning.
The directory structure is as follows:
 
## [C++](https://github.com/Pk13055/machine-learning/blob/master/C++)
  - <Failed C++ program to scale gradient descent>
  - Other misc C++ and C language code

## [Octave](https://github.com/Pk13055/machine-learning/blob/master/0Octave)
  - Octave implementation of the gradient descent provblems as prescribed by the course.
  - Mostly vectorized implementations with graphing and visual cues as well.

## [Python](https://github.com/Pk13055/machine-learning/blob/master/Scripts/RUN.md)
- [Normalization.py](https://github.com/Pk13055/machine-learning/blob/master/Scripts/normalization.py)
	- This file takes a dataset and applies feature scaling and mean normalization
	- Call the script as ./normalization.py ` <input file name> ` `<output file name>`
	- The input file must be a n + 1 feature followed by value, |n separated file
	- Output file will be a mean stdev valued file, followed by the dataset

- [data_set.py](https://github.com/Pk13055/machine-learning/blob/master/Scripts/dataset_gen.py) 
	- This file generates the dataset for the multivaritive linear regression problems
   - Enter ./data_set.py ` n value ` ` m value ` filename `

- [grad-descent.py](https://github.com/Pk13055/machine-learning/blob/master/Scripts/grad-descent.py)
   - full fledged implementation of multivaritive linear regression based gradient descent
   - It works with both classification as well as regression problems.
   - Features include:
   	- Auto detection of type of dataset
	- dynamic alpha calculation, with customizable limits.
	- Normalization support
	- Provision for regularized datasets as well.
