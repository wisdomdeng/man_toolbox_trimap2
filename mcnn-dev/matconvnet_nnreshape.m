function [Y] = matconvnet_nnreshape(X, winsize, dzdy)

%%  X is a single array of dimension W x H x D x N,
%   where D = winsize(1) x winsize(2) x winsize(3).
%   dzdy is a single array of dimension winsize(1) x winsize(2) x winsize(3) x N.

if nargin <= 2 || isempty(dzdy)
  Y = reshape(X, [winsize(1) winsize(2) winsize(3) size(X,4)]);
else
  dzdx = reshape(dzdy, size(X));
  Y = dzdx;
end

end
