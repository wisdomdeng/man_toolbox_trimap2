function [net, state] = dagcnn_param_init(net, opts, mode)

np = numel(net.params);
for i = 1:np,
  state.params(i).mgrad = 0 * net.params(i).value;
  if strcmp(opts.solver, 'adam'),
    state.params(i).vgrad = 0 * net.params(i).value;
  end
end

if opts.useGPU,
  net.move('gpu');
  for i = 1:np,
    state.params(i).mgrad = gpuArray(state.params(i).mgrad);
    if strcmp(opts.solver, 'adam'),
      state.params(i).vgrad = gpuArray(state.params(i).vgrad);
    end
  end
end

end
