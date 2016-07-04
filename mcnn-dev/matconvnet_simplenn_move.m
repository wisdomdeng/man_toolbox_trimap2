function net = matconvnet_simplenn_move(net, destination)

switch destination
  case 'gpu', moveop = @(x) gpuArray(x);
  case 'cpu', moveop = @(x) gather(x);
  otherwise, error('Unknown destination ''%s''.', destination);
end

for l = 1:numel(net.layers)
  switch net.layers{l}.type
  case {'conv', 'convt'}
    optionf = {'filters','biases','mgrad.filters','vgrad.filters','mgrad.biases','vgrad.biases'};
  otherwise 
    optionf = [];
  end

  for f = optionf
    f = char(f);
    if isfield(net.layers{l},f)
      net.layers{l}.(f) = moveop(net.layers{l}.(f));
    end
  end

end

end
