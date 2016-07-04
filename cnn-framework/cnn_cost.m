function [cost, res] = cnn_cost(data, groundtruth, net, res, errorType)

gpuMode = isa(data, 'gpuArray');
one = single(1);
if gpuMode,
  one = gpuArray(one);
end

cost = 0;

net.layers{end}.class = groundtruth;
res = matconvnet_simplenn(net, data, one, res);

cost = cost + sum(res(end).x(:));

end
