function net = vae_param_update_adam(opts, net, res, lr, beta1, beta2, epsilon, ii, flag_dec)
%% TODO: only supports simplenn

nl = numel(net.layers);

if ii <  opts.adam.decayround
  lr = lr * sqrt(1 - (1 - beta2).^ii) ./ (1 - (1 - beta1).^ii);
end

if flag_dec && ii > 4000 && isfield(opts,'lr_std'),
  std_lr = opts.lr_std * lr;
else
  std_lr = lr;
end

for l = 1:nl
  switch net.layers{l}.type,
  case {'conv','conv_part','conv_olpart', 'convt', 'fc_mix', 'fc_mix2','stream_split_mask', 'stream_split_std', 'stream_split_std_mask'}
    % weightDecay
      grad_filters = res(l).filters - net.layers{l}.filters * opts.weightDecay;
      grad_biases = res(l).biases - net.layers{l}.biases * opts.weightDecay;

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
      net.layers{l}.filters = net.layers{l}.filters + cpu2gpu_copy(lr * delta_filters, opts.useGPU);
      net.layers{l}.biases = net.layers{l}.biases + cpu2gpu_copy(lr * delta_biases, opts.useGPU);
  
  case {'early_split', 'conv_streams', 'conv_gaussian', 'convt_gaussian', 'fc_mix_gaussian','fc_mix2_gaussian'}
    % weightDecay
      grad_filters = res(l).filters - net.layers{l}.filters * opts.weightDecay;
      grad_biases = res(l).biases - net.layers{l}.biases * opts.weightDecay;
      grad_filters_std = res(l).filters_std - net.layers{l}.filters_std * opts.weightDecay;
      grad_biases_std = res(l).biases_std - net.layers{l}.biases_std * opts.weightDecay;

    % mgrad
      net.layers{l}.mgrad.filters = net.layers{l}.mgrad.filters + beta1 * (grad_filters - net.layers{l}.mgrad.filters);
      net.layers{l}.mgrad.biases = net.layers{l}.mgrad.biases + beta1 * (grad_biases - net.layers{l}.mgrad.biases);
      net.layers{l}.mgrad.filters_std = net.layers{l}.mgrad.filters_std + beta1 * (grad_filters_std - net.layers{l}.mgrad.filters_std);
      net.layers{l}.mgrad.biases_std = net.layers{l}.mgrad.biases_std + beta1 * (grad_biases_std - net.layers{l}.mgrad.biases_std);

    % vgrad
      net.layers{l}.vgrad.filters = net.layers{l}.vgrad.filters + beta2 * (grad_filters.^2 - net.layers{l}.vgrad.filters);
      net.layers{l}.vgrad.biases = net.layers{l}.vgrad.biases + beta2 * (grad_biases.^2 - net.layers{l}.vgrad.biases);
      net.layers{l}.vgrad.filters_std = net.layers{l}.vgrad.filters_std + beta2 * (grad_filters_std.^2 - net.layers{l}.vgrad.filters_std);
      net.layers{l}.vgrad.biases_std = net.layers{l}.vgrad.biases_std + beta2 * (grad_biases_std.^2 - net.layers{l}.vgrad.biases_std);

    % delta
      delta_filters = (net.layers{l}.mgrad.filters) ./ (sqrt(net.layers{l}.vgrad.filters) + epsilon);
      delta_biases = (net.layers{l}.mgrad.biases) ./ (sqrt(net.layers{l}.vgrad.biases) + epsilon);
      delta_filters_std = (net.layers{l}.mgrad.filters_std) ./ (sqrt(net.layers{l}.vgrad.filters_std) + epsilon);
      delta_biases_std = (net.layers{l}.mgrad.biases_std) ./ (sqrt(net.layers{l}.vgrad.biases_std) + epsilon);

    % update
      net.layers{l}.filters = net.layers{l}.filters + cpu2gpu_copy(lr * delta_filters, opts.useGPU);
      net.layers{l}.biases = net.layers{l}.biases + cpu2gpu_copy(lr * delta_biases, opts.useGPU);
      net.layers{l}.filters_std = net.layers{l}.filters_std + cpu2gpu_copy(std_lr * delta_filters_std, opts.useGPU);
      net.layers{l}.biases_std = net.layers{l}.biases_std + cpu2gpu_copy(std_lr * delta_biases_std, opts.useGPU);
  end
end

if opts.debug
  for l = 1:nl
    tt1 = res(l+nin).x;
    fprintf('layer %d, f: %g %g\n', l, min(tt1(:)), max(tt1(:)));
    tt2 = res(l+nin).dzdx;
    fprintf('layer %d, grad: %g %g\n', l, min(tt2(:)), max(tt2(:)));
  end
  pause;
end

end
