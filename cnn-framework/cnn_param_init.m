function net = cnn_param_init(net, opts, mode)

nl = numel(net.layers);

for i = 1:nl,
  switch net.layers{i}.type
  case {'conv','convt'}
    net.layers{i}.mgrad.filters = 0 * net.layers{i}.filters;
    net.layers{i}.mgrad.biases = 0 * net.layers{i}.biases;

    if strcmp(opts.solver, 'adam'),
      net.layers{i}.vgrad.filters = 0 * net.layers{i}.filters;
      net.layers{i}.vgrad.biases = 0 * net.layers{i}.biases;
    end
  end
end

if opts.useGPU,
  net = matconvnet_simplenn_move(net, 'gpu');
end

end
