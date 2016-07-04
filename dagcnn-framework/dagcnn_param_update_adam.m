function [net, state] = dagcnn_param_update_adam(opts, net, state, lr, beta1, beta2, epsilon, ii)

np = numel(net.params);

if ii < opts.adam.decayround,
  lr = lr * sqrt(1 - (1 - beta2).^ii) ./ (1 - (1 - beta1).^ii);
end

for i = 1:np,
  grad_params = net.params(i).der - net.params(i).value * opts.weightDecay;

  % mgrad
  state.params(i).mgrad = state.params(i).mgrad + (beta1) * (grad_params - state.params(i).mgrad);

  % vgrad
  state.params(i).vgrad = state.params(i).vgrad + (beta2) * (grad_params.^2 - state.params(i).vgrad);

  % delta
  delta_params = state.params(i).mgrad ./ (sqrt(state.params(i).vgrad) + epsilon);

  %fprintf('%.3f\n', mean(delta_params(:)));
  % update
  net.params(i).value = net.params(i).value - cpu2gpu_copy(lr * delta_params, opts.useGPU);

end

end
