# K - Means Unsupervised Learning Algorithm

## Files

- ` config.py `: This file contains all the config variables used to set various params
of the algorithm runtime. 
- `helper.py`: This contains all the helper methods used as part of the main functions of k-means
but not necessarily relating to k-means.
- `k_means.py` : This is the main script file. It is *essential* to have numpy and/or matplotlib installed, to make
full use of the visualization done.


## How to run

- `k_means.py <filename> , [<K1>, [<K2, [ ... ]] `: Broken down:
	- `<filename>`: The name of the dataset file. It **has** to follow the format _x11 x12 x13 ... xn \n ... \n xm1 xm2 ... xmn_
	- `<K*>`: Optional paramters to state the number of clusters. (_default is 2_)

## Documentation

- This implementation of K-means is a full-fledged _smart_ implementation. You have an option to specify multiple (_or 1 or 0 _) number of clusters and K - means will be run against all different combinations. 
- In addition to this, within induvidual cluster choices, this algorithm will run K - means multiple times, with different initial values to avoid local optima (_in this case minima_).
- The final returned value of the centroids will be the best possible value across different _not-so-random_ random initializations across different cluster numbers.
- Extensive use of `matplotlib` so as to help visualize the data, with _color-coordinated_, cluster-grouped data representations.
