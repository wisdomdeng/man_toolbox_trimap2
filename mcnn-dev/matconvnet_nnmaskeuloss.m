function Y = matconvnet_nnmaskeuloss(X, gt, mask, dzdy)
% MATCONVNET_NNMASKEULOSS  CNN Masked Euclidean Loss

sz = [size(X,1) size(X,2) size(X,3) size(X,4)];

n = sz(1) * sz(2);
if nargin <= 3
  intm = bsxfun(@times, bsxfun(@minus, X, gt), mask);
  Y = sum(intm(:).^2) / n;
else
  Y = 2 * bsxfun(@times, X - gt, mask) * (dzdy / n);
end

end
