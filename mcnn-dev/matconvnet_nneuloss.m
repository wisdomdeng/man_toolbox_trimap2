function Y = matconvnet_nneuloss(X, gt, dzdy)
% MATCONVNET_NNRMSELOSS  CNN euclidean loss

sz = [size(X,1) size(X,2) size(X,3) size(X,4)];

n = sz(1) * sz(2);
if nargin <= 2
  Y = sum((X(:) - gt(:)).^2) / n;
else
  Y = 2 * (X - gt) * (dzdy / n);
end

end
