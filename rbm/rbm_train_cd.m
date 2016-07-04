function [net, info] = rbm_train_cd(opts, net, imdb, getBatch)
%% binary RBM training using KCD

if ~exist(opts.expDir), mkdir(opts.expDir); end

opts.train = find(imdb.images.set == 1);
opts.val = find(imdb.images.set == 2);
% random seed
rng('default');

% convert to jacket (gpu) variables
net.vishid = cpu2gpu_copy(net.vishid, opts.useGPU);
net.vbias = cpu2gpu_copy(net.vbias, opts.useGPU);
net.hbias = cpu2gpu_copy(net.hbias, opts.useGPU);

net.vishidMMT = cpu2gpu_copy(zeros(size(net.vishid)), opts.useGPU);
net.vbiasMMT = cpu2gpu_copy(zeros(size(net.vbias)), opts.useGPU);
net.hbiasMMT = cpu2gpu_copy(zeros(size(net.hbias)), opts.useGPU);

%%
train = opts.train(randperm(numel(opts.train)));
val = opts.val;

info = struct;
info = rbm_info_init(opts, info, 0);

lr = 0;

res = [];
res2 = [];

%% Persistent CD initialization
if opts.usePCD
  res = rbm_pcd_init(net, opts);
  res2 = res;
end

modelPath = fullfile(opts.expPath, 'net-epoch-%d.mat');
modelFigPath = fullfile(opts.expPath, 'net-train.pdf');

for epoch = opts.numEpochs:-100:0,
  if opts.continue,
    if exist(sprintf(modelPath, epoch), 'file'), break; end
  end
end

if epoch ~= 0,
  fprintf('resuming by loading epoch %d\n', epoch);
  load(sprintf(modelPath, epoch), 'net', 'info');
  start_epoch = epoch+1;
else
  start_epoch = 1;
end

for epoch=start_epoch:opts.numEpochs,
  
  prevLr = lr;
  lr = opts.learningRate(min(epoch, numel(opts.learningRate)));
  momentum = opts.momentum(min(epoch, numel(opts.momentum)));

  % fast-forward to where we stopped
  modelPath = fullfile(opts.expPath, 'net-epoch-%d.mat');
  modelFigPath = fullfile(opts.expPath, 'net-train.pdf');

  %if opts.continue
  %  if exist(sprintf(modelPath, epoch), 'file'), continue; end
  %  if epoch > 1
  %    fprintf('resuming by loading epoch %d\n', epoch-1);
  %    load(sprintf(modelPath, epoch-1), 'net', 'info');
  %  end
  %end

  info = rbm_info_init(opts, info, 1);
  % reset momentum if needed
  if prevLr ~= lr
    fprintf('learning rate changed (%f --> %f): resetting momentum\n', prevLr, lr);
    net.vishidMMT = 0 * net.vishidMMT;
    net.vbiasMMT = 0 * net.vbiasMMT;
    net.hbiasMMT = 0 * net.hbiasMMT;
  end

  % mini-batch mode -- training
  for t = 1:opts.batchSize:numel(train),
    batch = train(t:min(t+opts.batchSize-1, numel(train)));
    batch_time = tic;

    fprintf('training: epoch %02d: processing batch %3d of %3d (mmt = %g, lr = %g)...\n',...
      epoch, fix(t/opts.batchSize)+1, ceil(numel(train)/opts.batchSize), momentum, lr / (1 + epoch * opts.eps_decay));

    im = getBatch(imdb, batch);

    % im:      numvis x N
    % vishid:  numvis x numhid
    % visbias: numvis x 1
    % hidbias: numhid x 1

    %% contrastive divergence steps
    res = rbm_simplenn(net, im, res, opts, 'backward');
    
    dzdw = zeros(size(net.vishid));
    dzdhbias_reg = zeros(size(net.hbias));
    dzdw_reg = opts.weightDecay * net.vishid;
    % sparsity constraint
    if opts.sparsity.reg > 0
      switch opts.sparsity.mode
      case 'exact'
        hmh = res.hprob.pos.*(1-res.hprob.pos);
        mh = sum(res.hprob.pos, 2)/opts.batchSize;
        mh = min(max(mh, 1e-6), 1-1e-6); % numerical stability
        mhtmp = -opts.sparsity.tgt./mh + (1-opts.sparsity.tgt)./(1-mh);
        
        dobj = opts.sparsity.reg*bsxfun(@times, mhtmp, hmh)/opts.batchSize;
        
        dzdw_reg = dzdw_reg + single(im)*gather(dobj');
        dzdhbias_reg = dzdhbias_reg + gather(sum(dobj, 2));

      case 'approx'
        %% TODO
      end
    end

    % update parameters
    net.vishidMMT = momentum * net.vishidMMT + ...
      lr / (1 + epoch * opts.eps_decay) * (res.dzdw - dzdw_reg);
    net.hbiasMMT = momentum * net.hbiasMMT + ...
      lr / (1 + epoch * opts.eps_decay) * (res.dzdhbias - dzdhbias_reg);
    net.vbiasMMT = momentum * net.vbiasMMT + ...
      lr / (1 + epoch * opts.eps_decay) * res.dzdvbias;

    net.vishid = net.vishid + net.vishidMMT;
    net.hbias = net.hbias + net.hbiasMMT;
    net.vbias = net.vbias + net.vbiasMMT;

    batch_time = toc(batch_time);
    speed = numel(batch)/batch_time;
    info.train = updateError(opts, info.train, net, res, batch_time);
    
    %if opts.verbose,
      fprintf(' %.2f s (%.1f images/s)', batch_time, speed);
      n = t + numel(batch) - 1;
      fprintf(' err %.3f sparsity %.2f saturation %.2f', info.train.error(end)/n, ...
        info.train.sparsity(end)/n*100, info.train.saturation(end)/n*100);
      fprintf('\n');
    %end

  end

  % mini-batch mode -- validation
  for t = 1:opts.batchSize:numel(val),
      
    batch = val(t:min(t+opts.batchSize-1, numel(val)));
    batch_time = tic;

    fprintf('validation: epoch %02d: processing batch %3d of %3d (mmt = %g, lr = %g)...\n',...
      epoch, fix(t/opts.batchSize)+1, ceil(numel(val)/opts.batchSize), momentum, lr / (1 + epoch * opts.eps_decay));

    im = getBatch(imdb, batch);

    % im:      numvis x N
    % vishid:  numvis x numhid
    % visbias: numvis x 1
    % hidbias: numhid x 1

    %% contrastive divergence steps
    res2 = rbm_simplenn(net, im, res2, opts, 'forward');

    batch_time = toc(batch_time);
    speed = numel(batch)/batch_time;
    info.val = updateError(opts, info.val, net, res2, batch_time);
    
    %if opts.verbose,
      fprintf(' %.2f s (%.1f images/s)', batch_time, speed);
      n = t + numel(batch) - 1;
      fprintf(' err %.3f sparsity %.2f saturation %.2f', info.val.error(end)/n, ...
      info.val.sparsity(end)/n*100, info.val.saturation(end)/n*100);
      fprintf('\n');
    %end
  end

  info.numtrain = numel(train);
  info.numval = numel(val);
  [info, opts] = rbm_drawcurve(info, opts, epoch);
  if mod(epoch, 100) == 0
    save(sprintf(modelPath, epoch),'net','info');
  end

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
