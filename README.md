# Machine Learning Journey

This repo is all the stuff (extra) that I've done on my journey in machine learning.
The directory structure is as follows:
+ 
| 
|___+ C++
\  \
\  + <Failed C++ program to scale gradient descent>
\  \
\  + Other misc C++ and C language code
\
\
\__+ [Scripts](https://github.com/Pk13055/machine-learning/blob/master/Scripts/RUN.md)
   \
   + Normalization.py
   \	- This file takes a dataset and applies feature scaling and mean normalization
   \	- Call the script as ./normalization.py ` <input file name> ` `<output file name>`
   \	- The input file must be a n + 1 feature followed by value, |n separated file
   \	- Output file will be a mean stdev valued file, followed by the dataset
   \
   + data_set.py 
   \	- This file generates the dataset for the multivaritive linear regression problems
   \    - Enter ./data_set.py ` n value ` ` m value ` filename `
   \
   + grad.py
   \	- this is a simple working implementation of the univaritive linear regression based gradient descent algorithm
   \    - it works well for a learning rate of 0.01 and a dataset m size of <= 13
   \
   + grad-descent.py
   \    - full fledged implementation of multivaritive linear regression based gradient descent
