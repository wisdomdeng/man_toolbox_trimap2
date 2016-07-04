function [net, info] = cnn_train_adam(opts, net, imdb, getBatch, updateError)

%imdb.images.data = bsxfun(@minus, imdb.images.data, mean(imdb.images.data, 4));

if ~exist(opts.expDir), mkdir(opts.expDir); end

opts.train = find(imdb.set == 1);
opts.val = find(imdb.set == 2); 

%% train and validation
info = struct;
info = cnn_info_init(opts, info, 0);

rng(0);
net = cnn_param_init(net, opts, 'init');

lr = 0;
beta1 = opts.adam.beta1;
beta2 = opts.adam.beta2;
epsilon = opts.adam.epsilon;

res = [];
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
    load(sprintf('%s/net-epoch-%d.mat', opts.expPath, epoch_start));
    fprintf('resuming by loading epoch %d\n', epoch_start);
  end

  for epoch = 1:epoch_start,
    prevLr = lr;
    lr = opts.adam.learningRate(min(epoch, numel(opts.adam.learningRate)));
    if epoch > nround, % TODO
      lr = lr / (1 + opts.adam.lr_decay * (epoch-nround));
    end
    ii = ii + numel(1:opts.batchSize:numel(opts.train));
  end
end

for epoch=epoch_start+1:opts.numEpochs
  prevLr = lr ;
  lr = opts.adam.learningRate(min(epoch, numel(opts.adam.learningRate))) ;
  if epoch > nround, % TODO
    lr = lr / (1 + opts.adam.lr_decay * (epoch-nround));
  end

  % fast-forward to where we stopped
  modelPath = fullfile(opts.expPath, 'net-epoch-%d.mat');
  modelFigPath = fullfile(opts.expPath, 'net-train.pdf');
  %modelPath
  train = opts.train(randperm(numel(opts.train)));
  val = opts.val;

  info = cnn_info_init(opts, info, 1);

  %% mini-batch mode -- training
  for t = 1:opts.batchSize:numel(train)
    batch = train(t:min(t+opts.batchSize-1, numel(train)));
    batch_time = tic;
    %% TODO
    fprintf('training: epoch %02d: processing batch %3d of %3d (lr = %g)...\n', epoch, ...
            fix(t/opts.batchSize)+1, ceil(numel(train)/opts.batchSize), lr) ;
    
    [data, groundtruth] = getBatch(imdb, batch, opts) ;
    
    net.layers{end}.class = groundtruth;
    [cost, res] = cnn_cost(data, groundtruth, net, res, opts.errorType);
    ii = ii + 1;
    
    net = cnn_param_update_adam(opts, net, res, lr, beta1, beta2, epsilon, ii);
    
    batch_time = toc(batch_time);
    speed = numel(batch)/batch_time;
    
    switch opts.errorType,
    case 'euloss'
      info.train = updateError(opts, info.train, net, res, batch_time);
    case {'binary','multiclass'}
      info.train = updateError(opts, info.train, net, res, batch_time);
    end

    fprintf(' %.2f s (%.1f images/s)', batch_time, speed);
    n = t + numel(batch) -1;
    switch opts.errorType
    case 'binary'
      fprintf(' err %.1f ap %.3f iou %.4f', info.train.error(end)/n*100, info.train.ap(end)/n*100, ...
        info.train.iou(end)/n*100);
    case 'multiclass'
      fprintf(' err %.3f cls_err %.3f', info.train.error(end)/n*100, info.train.cls_err(end)/n*100);
    case 'euloss'
      fprintf(' err %.3f', info.train.error(end)/n);
    end
    fprintf('\n');

  end 
  
  %% mini-batch mode -- validation
  for t = 1:opts.batchSize:numel(val)
    batch = val(t:min(t+opts.batchSize-1, numel(val)));
    batch_time = tic;
    fprintf('validation: epoch %02d: processing batch %3d of %3d ...\n', epoch, ...
       fix(t/opts.batchSize)+1, ceil(numel(val)/opts.batchSize)) ;
    
    [data, groundtruth] = getBatch(imdb, batch, opts);
    net.layers{end}.class = groundtruth;

    [cost, res] = cnn_cost(data, groundtruth, net, res, opts.errorType);
    
    %%
    batch_time = toc(batch_time);
    speed = numel(batch)/batch_time;
   
    switch opts.errorType,
    case 'euloss'
      info.val = updateError(opts, info.val, net, res, batch_time);
    case {'binary','multiclass'}
      info.val = updateError(opts, info.val, net, res, batch_time);
    end
 
    fprintf(' %.2f s (%.1f images/s)', batch_time, speed);
    n = t + numel(batch) - 1;
    switch opts.errorType,
    case 'multiclass'
      fprintf(' err %.3f cls_err %.3f', info.val.error(end)/n*100, info.val.cls_err(end)/n*100);
    case 'binary'
      fprintf(' err %.1f ap %.3f iou %.3f', info.val.error(end)/n*100, info.val.ap(end)/n*100, ...
        info.val.iou(end)/n*100);
    case 'euloss'
      fprintf(' err %.3f', info.val.error(end)/n);
    end
    fprintf('\n');
    
  end

  info.numtrain = numel(train);
  info.numval = numel(val);
  [info, opts] = cnn_drawcurve(info, opts, epoch);

  if mod(epoch, opts.saveround) == 0,
    save(sprintf(modelPath, epoch), 'net', 'info');
  end
    
  if opts.verbose,
    drawnow;
    print(1, modelFigPath, '-dpdf');
  end

end % end of for

modelPath = fullfile(opts.expPath, 'net-epoch-%d.mat');
load(sprintf(modelPath, opts.numEpochs), 'net', 'info');

end

