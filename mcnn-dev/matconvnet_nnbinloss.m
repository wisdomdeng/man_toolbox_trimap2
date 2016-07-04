function Y = matconvnet_nnbinloss(X, c, dzdy)
%% log-loss for binary classification
%% X: H x W x 1 x N
%% c: H x W x 1 x N
%% dzdy: scalar

%% binloss = - 1/N * (c_n * log(sigmoid(x_n)) + (1 - c_n) * log(1-sigmoid( x_n))i)
c = 2 - c;

n = size(X,1) * size(X,2);
if nargin <= 2,
  Y = - sum(c(:).*log(sigmoid(X(:))) + (1.-c(:)).*log(1.-sigmoid(X(:)))) / n;
else
  dzdx = -(c-sigmoid(X)) .* (dzdy/n);
  Y = dzdx;
end

end

