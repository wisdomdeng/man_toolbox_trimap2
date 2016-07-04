function net = condrbm_params_init(opts, data)

net = struct;
net.condvis = opts.weightInit * randn(opts.numcond, opts.numvis);
net.condhid = opts.weightInit * randn(opts.numcond, opts.numhid);
net.vishid = opts.weightInit * randn(opts.numvis, opts.numhid);
net.vbias = arcsigm(clip(mean(data, 2))); %% TODO??
net.hbias = zeros(opts.numhid, 1);

end
