function res = rbm_sample(net, opts, N)

%% setup
res = struct;

res.hprob.pos = cpu2gpu_copy(rand(opts.numhid, N) < opts.sparsity.tgt, opts.useGPU);

upfactor = opts.upfactor;
downfactor = opts.downfactor;

vishid_bu = upfactor*net.vishid;
vishid_td = downfactor*net.vishid;

res.v.pos = sigmoid(bsxfun(@plus, vishid_td*res.hprob.pos, net.vbias));

end
