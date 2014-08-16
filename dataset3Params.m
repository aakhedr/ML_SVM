function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
models = [.01, .03, .1, .3, 1, 3, 10, 30];
errors = [];
for i = 1:length(models)
	for j = 1:length(models)
		model = svmTrain(X, y, models(i), @(x1, x2) gaussianKernel(x1, x2, models(j)), 1e-3, 20);
		
		predictions = svmPredict(model, Xval);
		errors = [errors; mean(double(predictions ~= yval)) models(i) models(j)];
	end
end
[error, index] = min(errors(:,1));
C = errors(index,2); sigma = errors(index,3);
% =========================================================================
end
