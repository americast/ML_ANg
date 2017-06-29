function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
new_y=(sigmoid(Theta1*X'))';
m = size(new_y, 1);
new_y = [ones(m, 1) new_y];
% size(new_y)
new_y=sigmoid(Theta2*new_y');
% size(new_y)
max_row=max(new_y', [],2);
% size(max_row)
for c=1:size(new_y,2);
	for d=1:size(new_y,1);
		if max_row(c)==new_y(d,c);
			p(c)=d;
		end;
	end;
end;








% =========================================================================


end
