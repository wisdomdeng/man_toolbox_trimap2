function net = vae_simple_move(net, destination)

switch destination
  case 'gpu', moveop = @(x) gpuArray(x) ;
  case 'cpu', moveop = @(x) gather(x) ;
  otherwise, error('Unknown desitation ''%s''.', destination) ;
end
for l=1:numel(net.layers)
  switch net.layers{l}.type
    case {'early_split', 'conv_streams', 'conv_gaussian','convt_gaussian', 'fc_mix_gaussian','fc_mix2_gaussian'}
      optionf = {'filters', 'biases', 'filters_std', 'biases_std', ...
              'mgrad.filters','vgrad.filters','mgrad.biases','mgrad.biases', ...
              'mgrad.filters_std', 'vgrad.filters_std', 'mgrad.biases_std', 'mgrad.biases_std'};
    case {'conv', 'conv_part','conv_olpart','convt', 'fc_mix', 'fc_mix2', 'stream_split_mask', 'stream_split_std', 'stream_split_std_mask'}
      optionf = {'filters', 'biases', 'mgrad.filters', 'vgrad.filters', 'mgrad.biases', 'vgrad.biases'};
    
    otherwise
      optionf = [];
  end

  for f = optionf
    f = char(f) ;
    if isfield(net.layers{l}, f)
      net.layers{l}.(f) = moveop(net.layers{l}.(f)) ;
    end
  end
  
end
