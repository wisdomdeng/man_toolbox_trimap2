function [cost, grad] = matconvnet_cost_mini(theta, opts, net, im, labels, l2reg)

%% TODO: l2reg 

%% matconvnet_unroll_pars
net = matconvnet_unroll_pars(theta, net, opts.useGpu);

%% cpuArray --> gpuArray
if opts.useGpu
  %im = gpuArray(im);
  one = gpuArray(single(1));
else
  one = single(1);
end

res = [];
% backprop
net.layers{end}.class = labels;
res = matconvnet_simplenn(net, im, one, res, ...
    'conserveMemory', opts.conserveMemory, ...
    'sync', opts.sync);

M = size(im, 4);
%% net --> grad, cost
grad = [];
cost = 0;
for i = 1:numel(net.layers),
  if strcmp(net.layers{i}.type,'conv')
    grad_filters = double(gather(res(i).dzdw{1})) ./ M + l2reg*double(gather(net.layers{i}.filters));
    grad_bias = double(gather(res(i).dzdw{2})) ./ M; 
    grad = [grad; grad_filters(:); grad_bias(:)];

    cost = cost + 0.5*l2reg*sum(double(gather(net.layers{i}.filters(:))).^2);
  elseif strcmp(net.layers{i}.type, 'bias')
    grad_bias = double(gather(res(i).dzdw{2})) ./ M;
    grad = [grad; grad_bias(:)];

  end

end

cost = cost + sum(double(gather(res(end).x))) / M;

clear im;
clear net;

end
