function net = mcn_dagcnn_reset(net)

[net.vars.value] = deal([]);
[net.vars.der] = deal([]);
[net.params.der] = deal([]);
for l = 1:numel(net.layers)
  net.layers(l).block.reset();
end

end
