function [Y, dzdw] = matconvnet_nnbias(X, biases, dzdy)

%% X is a single array of dimension  H x W x D x N where (H, W) are the height and width of the map stack, D is the image depth (number of feature channels) and N the number of images in the stack

%% biases is a single array of dimension H x W x D 

%% Y is a single array of dimension H x W x D x N

%% dzdw is a single array of dimension H x W x D

%% dzdy is a single array of dimension H x W x D x N

dzdw = [];

if nargin <= 2 || isempty(dzdy)
  Y = bsxfun(@plus, X, biases);
  
else
  N = size(X, 4);
  dzdx = dzdy;
  dzdw = sum(dzdy, 4);
  Y = dzdx;
end


end
