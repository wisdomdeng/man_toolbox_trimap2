function net = cnn_param_update_adam(opts, net, res, lr, beta1, beta2, epsilon, ii)

nl = numel(net.layers);

if ii < opts.adam.decayround,
  lr = lr * sqrt(1 - (1 - beta2).^ii) ./ (1 - (1 - beta1).^ii);
end

for l = 1:nl,
  switch net.layers{l}.type
  case {'conv','convt'}
    grad_filters = res(l).dzdw{1} - net.layers{l}.filters * opts.weightDecay;
    grad_biases = res(l).dzdw{2} - net.layers{l}.biases * opts.weightDecay;

    % mgrad
    net.layers{l}.mgrad.filters = net.layers{l}.mgrad.filters + beta1 * (grad_filters - net.layers{l}.mgrad.filters);
    net.layers{l}.mgrad.biases = net.layers{l}.mgrad.biases + beta1 * (grad_biases - net.layers{l}.mgrad.biases);

    % vgrad
    net.layers{l}.vgrad.filters = net.layers{l}.vgrad.filters + beta2 * (grad_filters.^2 - net.layers{l}.vgrad.filters);
    net.layers{l}.vgrad.biases = net.layers{l}.vgrad.biases + beta2 * (grad_biases.^2 - net.layers{l}.vgrad.biases);

    % delta
    delta_filters = (net.layers{l}.mgrad.filters) ./ (sqrt(net.layers{l}.vgrad.filters) + epsilon);
    delta_biases = (net.layers{l}.mgrad.biases) ./ (sqrt(net.layers{l}.vgrad.biases) + epsilon);
   
    % update
    net.layers{l}.filters = net.layers{l}.filters - cpu2gpu_copy(lr * delta_filters, opts.useGPU);
    net.layers{l}.biases = net.layers{l}.biases - cpu2gpu_copy(lr * delta_biases, opts.useGPU);

  end
end

end % 
