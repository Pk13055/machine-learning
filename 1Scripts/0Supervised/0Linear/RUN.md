# How to run the various scripts

## Dataset generation

- To generate the dataset all you need to do is run the ` data_set.py ` file
	- ` ./better_data_set.py < output_file > < features > < records >`
	- ` ./data_set.py < #features > < #records > < output file > < threshold* > `

- The varous parts of the command:
	- features: This parameter is compulsory and is for the number of features you want in your regression model
		If you choose this parameter as 3 this will include features ` x0 ( = 1), x1, x2, x3 ` followed by the yi
	- records: This paramter if for how many datapoints you want in your dataset. Smaller datapoints lead to quicker conversion but larger sets give more accurate results
	- output file: It is basically the name of the output file to which you want to write the data.
	- threshold (optional): It defines the threshold upto which the upper lower should be taken as is.

## Normalization

- To generate the normalized dataset all you need to do is run the ` normalization.py ` file.
- ` ./normalization.py < input_data_set > < output_data_set> ` 
- The parameters for the command are:
	- input_data_set: This is the file from which to read the input data set (_by default searches for the dataset.txt file_)
	- output_data_set: This is the file to which to write the normalized data (_by default writes to normalized_data.txt file_)

## Gradient-Descent

- To generate the params and other features, run the following:
	- ` ./grad-descent.py <learning rate > < dataset file > <  normalize_data file > < timeout* >`
	- Example set : 
		- ` ./grad-descent.py 1.9682999999999997 dataset2.txt dataset2_normal.txt ` => normalized
	 	- `./grad-descent.py 0.0000000769 dataset2.txt ` => unormalized 
	 	- `./ grad-descent.py 1.5783797700000002 big_data.txt big_data_normal.txt 3000` => normalized big dataset _(this is a *75 feature*, *1000 example* dataset)_
- The parameters for this command are:
	- *learning rate*: The learning rate for your gradient descent algorithm.
	- *dataset file*: The dataset for which your algorithm will run.
	- *normalized data*: The normalized datset (_optional_) which helps in quicker convergance.
	- *timeout*: This will kill long processess. _(entered in seconds)_
- Following the command, you will run into the possible scenarios:
	- Program crunches for a while following which you get a prompt: This is the success case, you can query for cases
	- Program runs into a [nan, nan, ...] print cycle: This means your learning rate is far too large. Use a smaller learning rate and, if possible, use normalized data
	- Program proceeds with [float, float ..] value printing: Program is crunching, give it time

## Folder Tree
.
 * [config.py](./config.py)
 * [dataset_gen.py](./dataset_gen.py)
 * [grad-descent.py](./grad-descent.py)
 * [normalization.py](./normalization.py)
 * [RUN.md](./RUN.md)