function net = matconvnet_dagnn_move(net, destination)

switch destination
  case 'gpu', moveop = @(x) gpuArray(x) ;
  case 'cpu', moveop = @(x) gather(x) ;
  otherwise, error('Unknown desitation ''%s''.', destination) ;
end
for l=1:numel(net.layers)
  switch net.layers{l}.type
    case {'conv','tensorprod'}
      for f = {'filters', 'biases', 'filtersMomentum', 'biasesMomentum'}
        f = char(f) ;
        if isfield(net.layers{l}, f)
          net.layers{l}.(f) = moveop(net.layers{l}.(f)) ;
        end
      end
    otherwise
      % nothing to do ?
  end
end
