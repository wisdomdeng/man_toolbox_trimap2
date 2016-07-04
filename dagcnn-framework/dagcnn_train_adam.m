function [net, info] = dagcnn_train_adam(opts, net, imdb, getBatch, extractStats, updateError)

if ~exist(opts.expDir), mkdir(opts.expDir); end

opts.train = find(imdb.set == 1);
opts.val = find(imdb.set == 2);

%% train and validation
info = struct;
info = dagcnn_info_init(opts, info, 0);

rng(0);
[net, state] = dagcnn_param_init(net, opts, 'init');

lr = 0;
beta1 = opts.adam.beta1;
beta2 = opts.adam.beta2;
epsilon = opts.adam.epsilon;

one = cpu2gpu_copy(1.0, opts.useGPU);

numbatch = min(floor(min(numel(opts.train), 100000)/opts.batchSize), 20);
nround = ceil(opts.adam.decayround/numbatch);
ii = 0;

if ~opts.continue,
  epoch_start = 0;
else
  epoch_start = 0;
  while exist(sprintf('%s/net-epoch-%d.mat', opts.expPath, epoch_start+opts.saveround), 'file')
    epoch_start = epoch_start + opts.saveround;
  end
  if epoch_start > 0
    netStruct = load(sprintf('%s/net-epoch-%d.mat', opts.expPath, epoch_start));
    net = dagnn.DagNN.loadobj(netStruct);
    clear netStruct;
    state = load(sprintf('%s/state-epoch-%d.mat', opts.expPath, epoch_start));
    if opts.useGPU,
      net.move('gpu');
    end
  end

  for epoch = 1:epoch_start,
    prevLr = lr;
    lr = opts.adam.learningRate(min(epoch, numel(opts.adam.learningRate)));
    if epoch > nround, 
      lr = lr / (1 + opts.adam.lr_decay * (epoch - nround));
    end
    ii = ii + numel(1:opts.batchSize:numel(opts.train));
  end
end

for epoch = epoch_start+1:opts.numEpochs,
  prevLr = lr;
  lr = opts.adam.learningRate(min(epoch, numel(opts.adam.learningRate)));
  if epoch > nround, 
    lr = lr / (1 + opts.adam.lr_decay * (epoch - nround));
  end

  %% fast-forward to where we stopped
  
  train = opts.train(randperm(numel(opts.train)));
  val = opts.val;

  info = dagcnn_info_init(opts, info, 1);

  %% mini-batch mode --training
  for t = 1:opts.batchSize:numel(train),
    batch = train(t:min(t+opts.batchSize-1, numel(train)));
    batch_time = tic;
    fprintf('training: epoch %02d: processing batch %03d of %03d (lr = %g)...\n', ...
      epoch, fix(t/opts.batchSize)+1, ceil(numel(train)/opts.batchSize), lr);

    [inputs, derOutputs] = getBatch(imdb, batch, opts);
    %unique(inputs{4})
    net.eval(inputs, derOutputs)
    %unique(gather(net.vars(18).value))
     %fprintf('%d %d %d %d\n', sum(t_pred(:)==1), sum(t_pred(:)==2), sum(t_pred(:)==3), sum(t_pred(:)==4));
    stats = extractStats(net);
    ii = ii + 1;

    [net, state] = dagcnn_param_update_adam(opts, net, state, lr, beta1, beta2, epsilon, ii);

    batch_time = toc(batch_time);
    speed = numel(batch)/batch_time;

    switch opts.errorType,
    case {'binary','multiclass'}
      info.train = updateError(opts, info.train, stats, numel(batch), batch_time);
    end

    fprintf(' %.2f s (%.1f images/s)', batch_time, speed);
    n = t + numel(batch) - 1;
    switch opts.errorType
    case {'binary','multiclass'}
      fprintf(' err %.1f ap %.3f iou %.4f',info.train.error(end)/n*100, ...
        info.train.ap(end)/n*100, info.train.iou(end)/n*100);
    end
    fprintf('\n');
  end

  %% mini-batch mode -- validation
  for t = 1:opts.batchSize:numel(val),
    batch = train(t:min(t+opts.batchSize-1, numel(val)));
    batch_time = tic;
    fprintf('validation: epoch %02d: processing batch %03d of %03d (lr = %g)...\n', ...
      epoch, fix(t/opts.batchSize)+1, ceil(numel(val)/opts.batchSize), lr);

    [inputs,~] = getBatch(imdb, batch, opts);
    net.eval(inputs)
    stats = extractStats(net);
    batch_time = toc(batch_time);
    speed = numel(batch)/batch_time;

    switch opts.errorType,
    case {'binary','multiclass'}
      info.val = updateError(opts, info.val, stats, numel(batch), batch_time);
    end

    fprintf(' %.2f s (%.1f images/s)', batch_time, speed);
    n = t + numel(batch) - 1;
    switch opts.errorType
    case {'binary','multiclass'}
      fprintf(' err %.1f ap %.3f iou %.4f',info.val.error(end)/n*100, ...
        info.val.ap(end)/n*100, info.val.iou(end)/n*100);
    end
    fprintf('\n');
  end

  info.numtrain = numel(train);
  info.numval = numel(val);
  [info, opts] = dagcnn_drawcurve(info, opts, epoch);

  if mod(epoch, opts.saveround) == 0,
    modelPath = sprintf('%s/net-epoch-%d.mat', opts.expPath, epoch);
    statePath = sprintf('%s/state-epoch-%d.mat', opts.expPath, epoch);
    
    netStruct = net.saveobj();
    save(modelPath, '-struct', 'netStruct');
    clear netStruct;
    save(statePath, 'state', 'info');
    
    if opts.useGPU,
      net.move('gpu');
    end
  end

  if opts.verbose,
    drawnow;
    modelFigPath = sprintf('%s/net-train.pdf', opts.expPath);
    print(1, modelFigPath, '-dpdf');
  end

end % end of for

netStruct = load(sprintf('%s/net-epoch-%d.mat', opts.expPath, epoch_start+opts.saveround));
net = dagnn.DagNN.loadobj(netStruct);
clear netStruct;
 
end

