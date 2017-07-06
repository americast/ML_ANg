function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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

values= [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
model= svmTrain(X, y, values(1), @(x1, x2) gaussianKernel(x1, x2, values(1)));
predictions = svmPredict(model, Xval);
error_out=mean(double(predictions ~= yval));
c_min=1; s_min=1; found=0;
for i=1:length(values)
	if found==1
		break;
	end;
	for j=1:length(values)
		if i==1 && j==1
			continue;
		end;
		C=values(i);
		sigma=values(j);
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		predictions = svmPredict(model, Xval);
		error_here=mean(double(predictions ~= yval));
		if error_here<error_out
			c_min=C; s_min=sigma;
			error_out=error_here;
		end;
		if error_here==0
			found=1;
			break;
		end;
	end;
end;
C=c_min
sigma=s_min






% =========================================================================

end
