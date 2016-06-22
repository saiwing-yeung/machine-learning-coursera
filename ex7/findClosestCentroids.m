function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

m = size(X, 1);
num_n = size(X, 2);

X_rep = repmat(X, 1, K);
centroids_flat = centroids'(:)';
X_diff = X_rep - repmat(centroids_flat, m, 1);
X_diffsq = X_diff .^ 2;
dist = zeros(m, K);

for i = 1:K
cur_indice = (1:num_n) + num_n*(i-1);
dist(:, i) = sum(X_diffsq(:, cur_indice), 2);
end

[ min_val, min_idx ] = min(dist, [], 2);

idx = min_idx;

% =============================================================

end

