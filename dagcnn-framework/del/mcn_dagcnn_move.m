function net = mcn_dagcnn_move(net, device)

net = mcn_dagcnn_reset(net);
net.device = device;
switch device
case 'gpu'
  for i = 1:numel(net.params)
    net.params(i).value = gpuArray(net.params(i).value);
    if isfield(net.params(i), 'mgrad'),
      net.params(i).mgrad = gpuArray(net.params(i).mgrad);
    end
    if isfield(net.params(i), 'vgrad'),
      net.params(i).vgrad = gpuArray(net.params(i).vgrad);
    end
  end
case 'cpu'
  for i = 1:numel(net.params)
    net.params(i).value = gather(net.params(i).value);
    if isfield(net.params(i), 'mgrad'),
      net.params(i).mgrad = gather(net.params(i).mgrad);
    end
    if isfield(net.params(i), 'vgrad'),
      net.params(i).vgrad = gather(net.params(i).vgrad);
    end
  end
otherwise
  error('Device must be either ''cpu'' or ''gpu'' .');
end

for l = 1:numel(net.layers)
  net.layers(l).block.move(device);
end

end
