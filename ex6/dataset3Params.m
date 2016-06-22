function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
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


shared_param_seq = ([ 1; 3 ] * [10 .^ (-2:1)])(:);

C_vec = shared_param_seq;%0.3;
sigma_vec = shared_param_seq;

C_col =     repmat(C_vec,     1, size(sigma_vec, 1))'(:);
sigma_col = repmat(sigma_vec, size(C_vec, 1), 1) (:);
num_run = size(C_col, 1);
param_mat = [ C_col sigma_col zeros(num_run, 2)] ;

% Try different SVM Parameters here
for i = 1:num_run
disp(sprintf('\n*\n***    Run %d / %d\n*\n', i, num_run));
C = param_mat(i, 1);
sigma = param_mat(i, 2);
model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

p_t = svmPredict(model, X);
error_t = 1 - mean(double(p_t == y));
fprintf('C: %2.2f    sigma: %2.2f    Training Error: %3.5f\n', C, sigma, error_t);

p_v = svmPredict(model, Xval);
error_v = 1 - mean(double(p_v == yval));
fprintf('C: %2.2f    sigma: %2.2f    Validation Accuracy: %3.5f\n', C, sigma, error_v);

param_mat(i, 3) = error_t;
param_mat(i, 4) = error_v;%sum(model.w .^ 2);
%visualizeBoundary(X, y, model);
end

%param_mat(:, 4) = (param_mat(:, 1) .^ 2 + param_mat(:, 2) .^ 2);

min_params = min(param_mat(:, 4));
min_params_mat = param_mat(find(param_mat(:, 4) == min_params), :)

C_best = min_params_mat(1);
sigma_best = min_params_mat(2);

fprintf('\nBest parameters: C = %.2f sigma = %.2f\n', C_best, sigma_best);


C = C_best;
sigma = sigma_best;




% =========================================================================

end
