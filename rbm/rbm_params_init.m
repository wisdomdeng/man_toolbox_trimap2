function net = rbm_params_init(opts)

net = struct;

net.vishid = opts.weightInit * randn(opts.numvis, opts.numhid) / sqrt(opts.numvis);
net.vbias = zeros(opts.numvis, 1);
net.hbias = zeros(opts.numhid, 1);

end
