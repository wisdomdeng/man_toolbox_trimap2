function [cost, grad] = matconvnet_cost_dagnn(theta, opts, net, spec, din, dout, l2reg)

%
n = numel(net.layers);
nin = numel(din.layers);
nout = numel(dout.layers);

%
net = matconvnet_unroll_pars(theta, net, opts.useGPU);

res = [];
% backprop

%% TODO: layer specific "class"
for i = 1:nout,
  net.layers{n - nout + i}.class = dout.layers{i}.x;
end
res = matconvnet_dagnn(net, spec, din, dout, res, ...
  'conserveMemory', opts.conserveMemory, ...
  'sync', opts.sync);

% number of samples in a minbatch
M = size(din.layers{1}.x, 4);

% net --> grad, cost
grad = [];
cost = 0;

for i = 1:n,
  switch net.layers{i}.type
  case {'conv','tensorprod'}
    %% something one has to be careful at (subscript)
    grad_filters = double(gather(res(i+nin).dzdw{1})) ./ M + ...
      l2reg * double(gather(net.layers{i}.filters));
    grad_bias = double(gather(res(i+nin).dzdw{2})) ./ M;
    grad = [grad; grad_filters(:); grad_bias(:)];

    cost = cost + 0.5 * l2reg * sum(double(gather(net.layers{i}.filters(:)).^2));

  end

end

for i = nin+n-nout+1:nin+n
  %% TODO: weighted average
  cost = cost + sum(double(gather(res(i).x))) / M;
end

end
