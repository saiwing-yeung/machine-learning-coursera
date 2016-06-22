function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

%	my = @() nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);


XoneT = [ ones(1, size(X, 1)); X' ];
Z2 = Theta1 * XoneT;
A2 = sigmoid(Z2);
A2one = [ ones(1, size(A2, 2)); A2 ];

Z3 = Theta2 * A2one;
A3 = sigmoid(Z3);


Jsum = 0;
num_error = 0;
[ max_value, max_index ] = max(A3, [], 1);


for cur_m=1:m
	yvec = zeros(num_labels, 1);
	yvec(y(cur_m)) = 1;

	hx = A3(:, cur_m);
	%hx = zeros(num_labels, 1);
	%hx(max_index(cur_m)) = 1;

	%disp(sprintf('y = %d; pred = %d', y(cur_m), max_index(cur_m)));

	if (y(cur_m) ~= max_index(cur_m))
		num_error = num_error + 1;
	end

	Jsum = Jsum + sum(-yvec.*log(hx) - (1-yvec).*log(1-hx));
end

J = Jsum / m;

%disp(sprintf('num_error = %d', num_error));



% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

Triangle2 = zeros(num_labels, hidden_layer_size + 1);
Triangle1 = zeros(hidden_layer_size, input_layer_size + 1);


for cur_m=1:m
yvec = zeros(num_labels, 1);
yvec(y(cur_m)) = 1;
a3 = A3(:, cur_m);
a2 = A2(:, cur_m);
a1 = X(cur_m, :)';

%	L3 delta
delta3 = a3 - yvec;

%	L2 delta
z2 = [ 1; Z2(:, cur_m) ];
delta2 = Theta2' * delta3 .* sigmoidGradient(z2);

%	No L1 delta!!!
%	z1 = [ 1; a1 ];
delta2_nozero = delta2(2:end);
%	delta1 = Theta1' * delta2_nozero .* sigmoidGradient(z1);


Triangle2 = Triangle2 + delta3 * [1; a2]';
Triangle1 = Triangle1 + delta2_nozero * [1; a1]';
end

Theta2_grad = Triangle2 / m;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda .* Theta2(:, 2:end)) ./ m;

Theta1_grad = Triangle1 / m;
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda .* Theta1(:, 2:end)) ./ m;



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




const_term = lambda / (2*m);

Theta1_biasonly = Theta1(:, 2:end);
Theta2_biasonly = Theta2(:, 2:end);

reg_cost = sum(sum(Theta1_biasonly .^ 2)) + sum(sum(Theta2_biasonly .^ 2));

J = J + const_term * reg_cost;







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
