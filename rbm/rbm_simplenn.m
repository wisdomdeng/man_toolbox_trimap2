function res = rbm_simplenn(net, im, res, opts, mode)

%% setup
if strcmp(mode, 'forward')
  doder = false;
else
  doder = true;
end

N = size(im, 2);

if isempty(res),
  res = struct;
end

im = cpu2gpu_copy(im, opts.useGPU);

upfactor = opts.upfactor;
downfactor = opts.downfactor;

vishid_bu = upfactor*net.vishid;
vishid_td = downfactor*net.vishid;

% positive phase: sample h ~ P(h|v)
res.hprob.pos = sigmoid(bsxfun(@plus, vishid_bu'*im, net.hbias));

prediction = rand(size(res.hprob.pos)) < res.hprob.pos;
res.h.pos = cpu2gpu_copy(prediction, opts.useGPU);

% persistent CD
if ~opts.usePCD,
  res.h.neg = res.h.pos;
end

% negative phase: sample v' ~ P(v|h), sample h' ~ P(h|v')
for kcd = 1:opts.kcd,
  % negative data
  res.vprob.neg = sigmoid(bsxfun(@plus, vishid_bu*res.h.neg, net.vbias));
  prediction = rand(size(res.vprob.neg)) < res.vprob.neg;
  res.v.neg = cpu2gpu_copy(prediction, opts.useGPU);

  % hidden unit inference
  res.hprob.neg = sigmoid(bsxfun(@plus, vishid_td'*res.v.neg, net.hbias));
  prediction = rand(size(res.hprob.neg)) < res.hprob.neg;
  res.h.neg = cpu2gpu_copy(prediction, opts.useGPU);
end

% monitor reconstruction error
res.v.pos = sigmoid(bsxfun(@plus, vishid_td*res.hprob.pos, net.vbias));
res.err = norm(im - res.v.pos, 'fro');

% compute gradient
if doder,
  res.dzdw = (im*res.hprob.pos' - res.v.neg*res.hprob.neg') / N; % TODO
  res.dzdhbias = mean(res.hprob.pos, 2) - mean(res.hprob.neg, 2);
  res.dzdvbias = mean(im, 2) - mean(res.v.neg, 2);
end

end
