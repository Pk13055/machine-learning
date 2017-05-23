function j = J(x, y theta)

% X are the training examples
% y is the actual values of the dataset
% theta are the predicted values

m = size(x, 1);
skew_ans = x * theta;
partial_sum = (skew_ans - y) .^ 2;
j = 1/ (2 * m) * sum(partial_sum);

