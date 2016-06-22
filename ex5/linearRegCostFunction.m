function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

if size(find(X(:, 1) == 1), 1) == size(X, 1)
	%disp('X has a vector of one already');
	Xone = X;
else
	%disp('did not have one; adding the vector of one...');
	Xone = [ones(m, 1) X];
end

J_self = sum((Xone * theta - y) .^ 2) / (2*m);
J_reg = sum(theta(2:end) .^ 2) * lambda / (2*m);

J = J_self + J_reg;


grad_self = ((Xone * theta - y)' * Xone)' ./ m;
grad_self(2:end) = grad_self(2:end) + lambda / m .* theta(2:end);

grad = grad_self;


% =========================================================================

grad = grad(:);

end
