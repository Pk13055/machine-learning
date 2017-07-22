# K - Means Unsupervised Learning Algorithm

## Files

- `k_means.py` : This is the main script file. It is *essential* to have numpy and/or matplotlib installed, to make
full use of the visualization done.
- `anomaly_detect.py` : This script detects anomalies in your datasets with helpful plots (_uses matplotlib_) to visualize various errors and anomalies. (This script is based on a gaussian probability distribution)
- `image_compress.py`: This script compresses _most_ images.


## How to run

- `k_means.py <filename> , [<K1>, [<K2, [ ... ]] `: Broken down:
	- `<filename>`: The name of the dataset file. It **has** to follow the format _x11 x12 x13 ... xn \n ... \n xm1 xm2 ... xmn_
	- `<K*>`: Optional paramters to state the number of clusters. (_default is 2_)
- `./image_compress.py <filename> [<K1>, [<K2, [<K3>,  [... ]]]`
	- _filename_: The file path of the image that is to be compressed.
	- _[<K1>, [<K2, [<K3>,  [... ]]]_: A list of color values. The higher the value, the less the image gets compressed, however, the greater the clarity.
- `./anomaly_detect.py <filename> tolerance* `
	- _filename_: The filename of the dataset.
	- *tolerance*: The tolerance for the gaussian probability model. Higher tolerance implies a more forgiving system. (_optional paramter, default is 0.2_)

## Documentation

- This implementation of K-means is a full-fledged _smart_ implementation. You have an option to specify multiple (_or 1 or 0 _) number of clusters and K - means will be run against all different combinations. 
- In addition to this, within induvidual cluster choices, this algorithm will run K - means multiple times, with different initial values to avoid local optima (_in this case minima_).
- The final returned value of the centroids will be the best possible value across different _not-so-random_ random initializations across different cluster numbers.
- Extensive use of `matplotlib` so as to help visualize the data, with _color-coordinated_, cluster-grouped data representations.

## Project Tree
.
 * [Image_Data](./Image_Data)
 * [anomaly_detect.py](./anomaly_detect.py)
 * [k_means.py](./k_means.py)
 * [image_compress.py](./image_compress.py)
 * [RUN.md](./RUN.md)