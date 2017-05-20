# How to run the various scripts

## Dataset generation

- To generate the dataset all you need to do is run the ` data_set.py ` file
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
