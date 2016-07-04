function res = rbm_pcd_init(net, opts)

upfactor = opts.upfactor;
vishid_bu = upfactor * net.vishid;

res.vprob.neg = repmat(sigmoid(net.vbias), [1, opts.batchSize]);
prediction  = rand(size(res.vprob.neg)) < res.vprob.neg;
res.v.neg = cpu2gpu_copy(prediction, opts.useGPU);

res.hprob.neg = sigmoid(bsxfun(@plus, vishid_bu'*res.v.neg, net.hbias));
prediction = rand(size(res.hprob.neg)) < res.hprob.neg;
res.h.neg = cpu2gpu_copy(prediction, opts.useGPU);

end
