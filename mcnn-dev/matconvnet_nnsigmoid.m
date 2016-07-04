function y = matconvnet_nnsigmoid(x, dzdy)

if nargin <= 1 || isempty(dzdy)
  y = single(1)./ (single(1) + exp(-x));
else
  sx = single(1)./(single(1) + exp(-x));
  y = dzdy .* sx .* (single(1) - sx);
end
