function [net, info] = condrbm_train(opts, net, imdb, getBatch)
%% binary Cond RBM (unfactored) training using CD-PercLoss

if ~exist(opts.expDir), mkdir(opts.expDir); end

opts.train = find(imdb.images.set == 1);
opts.val = find(imdb.images.set == 2);
% random seed
rng('default');

% convert to jacket (gpu) variables
net.condvis = cpu2gpu_copy(net.condvis, opts.useGPU);
net.condhid = cpu2gpu_copy(net.condhid, opts.useGPU);

net.vishid = cpu2gpu_copy(net.vishid, opts.useGPU);
net.vbias = cpu2gpu_copy(net.vbias, opts.useGPU);
net.hbias = cpu2gpu_copy(net.hbias, opts.useGPU);

net.condvisMMT = cpu2gpu_copy(zeros(size(net.condvis)), opts.useGPU);
net.condhidMMT = cpu2gpu_copy(zeros(size(net.condhid)), opts.useGPU);

net.vishidMMT = cpu2gpu_copy(zeros(size(net.vishid)), opts.useGPU);
net.vbiasMMT = cpu2gpu_copy(zeros(size(net.vbias)), opts.useGPU);
net.hbiasMMT = cpu2gpu_copy(zeros(size(net.hbias)), opts.useGPU);

%%
train = opts.train(randperm(numel(opts.train)));
val = opts.val;

info = struct;
info = condrbm_info_init(opts, info, 0);

lr = 0;

res = [];

for epoch=1:opts.numEpochs,
  
  prevLr = lr;
  lr = opts.learningRate(min(epoch, numel(opts.learningRate)));
  momentum = opts.momentum(min(epoch, numel(opts.momentum)));

  % fast-forward to where we stopped
  modelPath = fullfile(opts.expPath, 'net-epoch-%d.mat');
  modelFigPath = fullfile(opts.expPath, 'net-train.pdf');

  if opts.continue
    if exist(sprintf(modelPath, epoch), 'file'), continue; end
    if epoch > 1
      fprintf('resuming by loading epoch %d\n', epoch-1);
      load(sprintf(modelPath, epoch-1), 'net', 'info');
    end
  end

  info = rbm_info_init(opts, info, 1);
  % reset momentum if needed
  if prevLr ~= lr
    fprintf('learning rate changed (%f --> %f): resetting momentum\n', prevLr, lr);
    net.condvisMMT = 0 * net.condvisMMT;
    net.condhidMMT = 0 * net.condhidMMT;
  
    net.vishidMMT = 0 * net.vishidMMT;
    net.vbiasMMT = 0 * net.vbiasMMT;
    net.hbiasMMT = 0 * net.hbiasMMT;
  end

  % mini-batch mode -- training
  for t = 1:opts.batchSize:numel(train),
    batch = train(t:min(t+opts.batchSize-1, numel(train)));
    batch_time = tic;

    fprintf('training: epoch %02d: processing batch %3d of %3d (mmt = %g, lr = %g)...\n',...
      epoch, fix(t/opts.batchSize)+1, ceil(numel(train)/opts.batchSize), momentum, lr);

    din = getBatch(imdb, batch);

    % din.data:     numvis x N
    % din.cdata;    numcond x N
    % 
    % condvis:      numcond x numvis
    % condhid:      numcond x numhid
    % vishid:       numvis x numhid
    % visbias:      numvis x 2
    % hidbias:      numhid x 2

    %% contrastive divergence steps
    res = condrbm_simplenn(net, din, res, opts, 'backward');
    % TODO

    % update parameters
    net.condvisMMT = momentum * net.condvisMMT + ...
      lr * (res.dzdwcv - opts.weightDecay * net.condvis);
    net.condhidMMT = momentum * net.condhidMMT + ...
      lr * (res.dzdwch - opts.weightDecay * net.condhid);
    
    net.vishidMMT = momentum * net.vishidMMT + ...
      lr * (res.dzdwvh - opts.weightDecay * net.vishid);
    net.hbiasMMT = momentum * net.hbiasMMT + ...
      lr * res.dzdhbias;
    net.vbiasMMT = momentum * net.vbiasMMT + ...
      lr * res.dzdvbias;

    net.condvis = net.condvis + net.condvisMMT;
    net.condhid = net.condhid + net.condhidMMT;
    
    net.vishid = net.vishid + net.vishidMMT;
    net.hbias = net.hbias + net.hbiasMMT;
    net.vbias = net.vbias + net.vbiasMMT;

    batch_time = toc(batch_time);
    speed = numel(batch)/batch_time;
    info.train = updateError(opts, info.train, net, res, batch_time);
    % TODO

    if opts.verbose,
      fprintf(' %.2f s (%.1f images/s)', batch_time, speed);
      n = t + numel(batch) - 1;
      fprintf(' err %.3f sparsity %.2f saturation %.2f', info.train.error(end)/n, ...
        info.train.sparsity(end)/n*100, info.train.saturation(end)/n*100);
      fprintf('\n');
    end

  end

  % mini-batch mode -- validation
  for t = 1:opts.batchSize:numel(val),
      
    batch = val(t:min(t+opts.batchSize-1, numel(val)));
    batch_time = tic;

    fprintf('validation: epoch %02d: processing batch %3d of %3d (mmt = %g, lr = %g)...\n',...
      epoch, fix(t/opts.batchSize)+1, ceil(numel(val)/opts.batchSize), momentum, lr);

    din = getBatch(imdb, batch);

    %% contrastive divergence steps
    res = condrbm_simplenn(net, din, res, opts, 'forward');

    batch_time = toc(batch_time);
    speed = numel(batch)/batch_time;
    info.val = updateError(opts, info.val, net, res, batch_time);
    
    if opts.verbose,
      fprintf(' %.2f s (%.1f images/s)', batch_time, speed);
      n = t + numel(batch) - 1;
      fprintf(' err %.3f sparsity %.2f saturation %.2f', info.val.error(end)/n, ...
      info.val.sparsity(end)/n*100, info.val.saturation(end)/n*100);
      fprintf('\n');
    end
  end

  info.numtrain = numel(train);
  info.numval = numel(val);
  [info, opts] = condrbm_drawcurve(info, opts, epoch);
  save(sprintf(modelPath, epoch),'net','info');
  if opts.verbose,
    drawnow;
    print(1, modelFigPath, '-dpdf');
  end

end % end of for

modelPath = fullfile(opts.expPath, 'net-epoch-%d.mat');
load(sprintf(modelPath, opts.numEpochs), 'net', 'info');

end

%% external function
function info = updateError(opts, info, net, res, speed)

prediction = gather(res.hprob.pos);
n = size(res.hprob.pos, 2);
info.speed(end) = info.speed(end) + speed;
switch opts.errorType
  case 'recon'
    info.error(end) = info.error(end) + res.err;
    info.sparsity(end) = info.sparsity(end) + mean(prediction(:)) * n;
    info.saturation(end) = info.saturation(end) + ...
      (mean(prediction(:) > 0.95) + mean(prediction(:) < 0.05)) * n;
end

end
