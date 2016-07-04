function net = vae_param_init(net, opts, mode)

nl = numel(net.layers);

% reset_mode
%if strcmp(mode, 'reset')
  for i = 1:nl
    switch net.layers{i}.type
    case {'conv','conv_part','conv_olpart','convt','fc_mix', 'fc_mix2','stream_split_mask', 'stream_split_std', 'stream_split_std_mask'}
      net.layers{i}.mgrad.filters = 0 * net.layers{i}.filters;
      net.layers{i}.mgrad.biases = 0 * net.layers{i}.biases;

      if strcmp(opts.solver, 'adam'),
        net.layers{i}.vgrad.filters = 0 * net.layers{i}.filters;
        net.layers{i}.vgrad.biases = 0 * net.layers{i}.biases;
      end
    case {'early_split', 'conv_streams', 'conv_gaussian','convt_gaussian',  'fc_mix_gaussian', 'fc_mix2_gaussian'}
      net.layers{i}.mgrad.filters = 0 * net.layers{i}.filters;
      net.layers{i}.mgrad.biases = 0 * net.layers{i}.biases;
      net.layers{i}.mgrad.filters_std = 0 * net.layers{i}.filters_std;
      net.layers{i}.mgrad.biases_std = 0 * net.layers{i}.biases_std;

      if strcmp(opts.solver, 'adam'),
        net.layers{i}.vgrad.filters = 0 * net.layers{i}.filters;
        net.layers{i}.vgrad.biases = 0 * net.layers{i}.biases;
        net.layers{i}.vgrad.filters_std = 0 * net.layers{i}.filters_std;
        net.layers{i}.vgrad.biases_std = 0 * net.layers{i}.biases_std;
      end

    end % end of case
  end
  if opts.useGPU 
    net = vae_simplenn_move(net, 'gpu');
  end
end
