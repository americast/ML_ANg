function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

for i=1:size(Y,1)
	for j=1:size(Y,2)
		if R(i,j)==1
			J+=(Theta(j,:)*X(i,:)' - Y(i,j))^2;
		end
	end
end
J/=2;
J+= (lambda /2) * (sum(sum(Theta.^2)) + sum(sum(X.^2)));

for i=1:size(X,1)
	idx = find(R(i, :)==1);
	Theta_temp = Theta(idx, :);
	Y_temp = Y(i, idx);
	X_grad (i, :) = (X(i, :) * Theta_temp' - Y_temp ) * Theta_temp;
end
X_grad .+= lambda .* X;
for i=1:size(Theta,1)
	idx = find(R(:, i)==1);
	X_temp = X(idx, :);
	Y_temp = Y(idx,i);
	temp =  (X_temp * Theta(i,:)' - Y_temp ) ;
	Theta_grad (i, :) = temp' * X_temp;
end
Theta_grad .+= lambda .* Theta;

	% for k=1:size(X,2)
	% 	sum=0;
	% 	for k=1:size(Theta,1)
	% 		for l=1:size(Theta,2)
	% 			if R(k,l)==1
	% 				sum+=(Theta(l,:)*X(k,:)' - Y(k,l))*Theta(l,k);
	% 			end
	% 		end
	% 	end
	% 	X_grad(i,k)=sum;








% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
