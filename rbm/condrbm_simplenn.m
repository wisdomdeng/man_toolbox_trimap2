function res = condrbm_simplenn(net, din, res, opts, mode)

%% setup
if strcmp(mode, 'forward')
  doder = false;
else
  doder = true;
end

N = size(din.data, 2);

if isempty(res),
  res = struct;
end

din.data = cpu2gpu_copy(din.data, opts.useGPU);
din.cdata = cpu2gpu_copy(din.cdata, opts.useGPU);

% positive phase: sample h ~ P(h|v,c)
res.hprob.pos = sigmoid(bsxfun(@plus, net.vishid'*din.data + net.condhid'*din.cdata, net.hbias));

prediction = rand(size(res.hprob.pos)) < res.hprob.pos;
res.h.pos = cpu2gpu_copy(prediction, opts.useGPU);

% negative phase (CD-percLoss)
% visible (init from all 0 hidden state)
res.vprob.neg = sigmoid(bsxfun(@plus, net.condvis'*din.cdata, net.vbias));
prediction = rand(size(res.vprob.neg)) < res.vprob.neg;
res.v.neg = cpu2gpu_copy(prediction, opts.useGPU);

% hidden
res.hprob.neg = sigmoid(bsxfun(@plus, net.condhid'*din.cdata + net.vishid'*din.data, net.hbias));
prediction = rand(size(res.hprob.neg)) < res.hprob.neg;
res.h.neg = cpu2gpu_copy(prediction, opts.useGPU);

% compute free energy
fey = condrbm_fey(net, din.cdata, res.v.neg);
res.v_best = res.v.neg;

% sample v' ~ P(v|h,c), sample h' ~ P(h|v',c)
for kcd = 1:opts.kcd,
  % visible
  res.vprob.neg = sigmoid(bsxfun(@plus, net.condvis'*din.cdata + net.vishid*res.h.neg, net.vbias));
  prediction = rand(size(res.vprob.neg)) < res.vprob.neg;
  res.v.neg = cpu2gpu_copy(prediction, opts.useGPU);

  % hidden
  res.hprob.neg = sigmoid(bsxfun(@plus, net.vishid'*res.v.neg + net.condhid'*din.cdata, net.hbias));
  prediction = rand(size(res.hprob.neg)) < res.hprob.neg;
  res.h.neg = cpu2gpu_copy(prediction, opts.useGPU);

  % compute free-energy and keep only if the free energy is lower
  cfey = condrbm_fey(net, din.cdata, res.v.neg);
  idx = cfey <= fey; % TODO??
  fey(idx) = cfey(idx);
  res.v_best(:,idx) = res.v.neg(:,idx);
end

% update visible state with lowest free energy
res.v.neg = res.v_best;
res.hprob.neg = sigmoid(bsxfun(@plus, net.vishid'*res.v.neg + net.condhid'*din.cdata, net.hbias));

% TODO: here
% monitor reconstruction error
res.v.pos = sigmoid(bsxfun(@plus, net.vishid*res.hprob.pos, net.vbias));
res.err = norm(din.data - res.v.pos, 'fro');

% compute gradient
if doder,
  res.dzdwvh = (din.data*res.hprob.pos' - res.v.neg*res.hprob.neg') / N; % TODO
  res.dzdwcv = (din.cdata*din.data' - din.cdata*res.v.neg') / N;
  res.dzdwch = (din.cdata*res.hprob.pos' - din.cdata*res.hprob.neg') / N;

  res.dzdhbias = mean(res.hprob.pos, 2) - mean(res.hprob.neg, 2);
  res.dzdvbias = mean(din.data, 2) - mean(res.v.neg, 2);   
end

end

%% -----------------------
% fey(v, c) = -log(sum_{h} exp(-E(v,h,c)))
%           = - c'Wv - v'b
%           = - sum_{j} log(1+exp(c'W_{j} + v'W_{j} + b{j})
% cdata, data, h: dim x batchSize
% fey           : 1 x batchSize
%% -----------------------
function fey = condrbm_fey(net, cdata, data, h)
  
% cond - vis, visbias
fey = -sum(cdata.*(net.condvis*data), 1);
fey = fey - net.vbias'*data;

if exist('hid', 'var') && ~isempty(hid),
  fey = fey - sum(cdata.*(net.condhid*h), 1);
  fey = fey - sum(data.*(net.vishid*h), 1);
  fey = fey - net.hbias'*h;
else
  fey = fey - sum(logexp(bsxfun(@plus, net.vishid'*data + net.condhid'*cdata, net.hbias)), 1);
end

end
