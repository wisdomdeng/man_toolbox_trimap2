function [net, info] = cnn_train_sgd(opts, net, imdb, getBatch, updateError)

imdb.images.data = bsxfun(@minus, imdb.images.data, mean(imdb.images.data, 4));

if ~exist(opts.expDir), mkdir(opts.expDir); end

opts.train = find(imdb.images.set == 1);
opts.val = find(imdb.images.set == 2); 

%% net init
for i = 1:numel(net.layers)
  if ~strcmp(net.layers{i}.type,'conv'), continue; end
  net.layers{i}.filtersMomentum = zeros('like', net.layers{i}.filters) ;
  net.layers{i}.biasesMomentum = zeros('like', net.layers{i}.biases) ;
  if ~isfield(net.layers{i}, 'filtersLearningRate')
    net.layers{i}.filtersLearningRate = 1;
  end
  if ~isfield(net.layers{i}, 'biasesLearningRate')
    net.layers{i}.biasesLearningRate = 1;
  end
  if ~isfield(net.layers{i}, 'filtersWeightDecay')
    net.layers{i}.filtersWeightDecay = 1;
  end
  if ~isfield(net.layers{i}, 'biasesWeightDecay')
    net.layers{i}.biasesWeightDecay = 1;
  end
end

if opts.useGPU
  net = vl_simplenn_move(net, 'gpu');
  for i = 1:numel(net.layers),
    if ~strcmp(net.layers{i}.type,'conv'), continue; end
    net.layers{i}.filtersMomentum = gpuArray(net.layers{i}.filtersMomentum) ;
    net.layers{i}.biasesMomentum = gpuArray(net.layers{i}.biasesMomentum) ;
  end
end

%% train and validation
info = struct;
info = cnn_info_init(opts, info, 0);

rng(0);

lr = 0;
res = [];
%if opts.useGpu,
%  one = gpuArray(single(1.0));
%else
%  one = single(1.0);
%end

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
end
 
one = cpu2gpu_copy(1.0, opts.useGPU);

for epoch=1:opts.numEpochs
  prevLr = lr ;
  lr = opts.sgd.learningRate(min(epoch, numel(opts.sgd.learningRate))) ;
  mmt = opts.sgd.momentum(min(epoch, numel(opts.sgd.momentum)));

  % fast-forward to where we stopped
  modelPath = fullfile(opts.expPath, 'net-epoch-%d.mat');
  modelFigPath = fullfile(opts.expPath, 'net-train.pdf');
  %modelPath
  train = opts.train(randperm(numel(opts.train)));
  val = opts.val;

  info = cnn_info_init(opts, info, 1);
  % reset momentum if needed
  if prevLr ~= lr
    fprintf('learning rate changed (%f --> %f): resetting momentum\n', prevLr, lr);
    for l = 1:numel(net.layers)
      if ~strcmp(net.layers{l}.type, 'conv'), continue; end
      net.layers{l}.filtersMomentum = 0 * net.layers{l}.filtersMomentum;
      net.layers{l}.biasesMomentum = 0 * net.layers{l}.biasesMomentum;
    end
  end

  %% mini-batch mode -- training
  for t = 1:opts.batchSize:numel(train)
    batch = train(t:min(t+opts.batchSize-1, numel(train)));
    batch_time = tic;
    %% TODO
    fprintf('training: epoch %02d: processing batch %3d of %3d (lr = %g, mmt = %g)...\n', epoch, ...
            fix(t/opts.batchSize)+1, ceil(numel(train)/opts.batchSize), lr, mmt) ;
    
    [im, labels] = getBatch(imdb, batch) ;
    
    %if opts.useGpu
    %  im = gpuArray(im);
    %end

    im = cpu2gpu_copy(im, opts.useGPU);
    
    %% Backprop Update
    net.layers{end}.class = labels;
    net_update = net;
    % nestrov momentum
    if opts.sgd.nestrov
      for l = 1:numel(net_update.layers)
        if ~strcmp(net_update.layers{l}.type, 'conv'), continue; end
        net_update.layers{l}.filters = net_update.layers{l}.filters + ...
          mmt * net_update.layers{l}.filtersMomentum;
        net_update.layers{l}.biases = net_update.layers{l}.biases + ...
          mmt * net_update.layers{l}.biasesMomentum;
      end
    end

    res = matconvnet_simplenn(net_update, im, one, res, ...
        'conserveMemory', opts.conserveMemory, ...
        'sync', opts.sync);

    clear net_update;
    % gradient step
    for l = 1:numel(net.layers)
      if ~strcmp(net.layers{l}.type, 'conv'), continue; end
      net.layers{l}.filtersMomentum = ...
        mmt * net.layers{l}.filtersMomentum ...
        - (lr * net.layers{l}.filtersLearningRate) * ...
        (opts.weightDecay * net.layers{l}.filtersWeightDecay) * net.layers{l}.filters ...
        - (lr * net.layers{l}.filtersLearningRate) / numel(batch) * res(l).dzdw{1};
      
      net.layers{l}.biasesMomentum = ...
      mmt * net.layers{l}.biasesMomentum ...
        - (lr * net.layers{l}.biasesLearningRate) * ...
        (opts.weightDecay * net.layers{l}.biasesWeightDecay) * net.layers{l}.biases ...
        - (lr * net.layers{l}.biasesLearningRate) / numel(batch) * res(l).dzdw{2};

      net.layers{l}.filters = net.layers{l}.filters + net.layers{l}.filtersMomentum;
      net.layers{l}.biases = net.layers{l}.biases + net.layers{l}.biasesMomentum;
    end

    if opts.sgd.nestrov
      res = matconvnet_simplenn(net, im, [], [], ...
        'conserveMemory', opts.conserveMemory, ...
        'sync', opts.sync);
    end

    batch_time = toc(batch_time);
    speed = numel(batch)/batch_time;
    info.train = updateError(opts, info.train, net, res, batch_time);
    
    fprintf(' %.2f s (%.1f images/s)', batch_time, speed);
    n = t + numel(batch) -1;
    switch opts.errorType
    case 'binary'
      fprintf(' err %.1f ap %.3f iou %.4f', info.train.error(end)/n*100, info.train.ap(end)/n*100, ...
        info.train.iou(end)/n*100);
    case 'multiclass'
      fprintf(' err %.3f cls_err %.3f', info.train.error(end)/n*100, info.train.cls_err(end)/n*100);
    end
    fprintf('\n');

  end 
  
  %% mini-batch mode -- validation
  for t = 1:opts.batchSize:numel(val)
    batch = val(t:min(t+opts.batchSize-1, numel(val)));
    batch_time = tic;
    fprintf('validation: epoch %02d: processing batch %3d of %3d ...\n', epoch, ...
       fix(t/opts.batchSize)+1, ceil(numel(val)/opts.batchSize)) ;
    [im, labels] = getBatch(imdb, batch);

    %if opts.useGpu
    %  im = gpuArray(im);
    %end
    im = cpu2gpu_copy(im, opts.useGPU);

    net.layers{end}.class = labels;
    res = matconvnet_simplenn(net, im, [], [], ...
        'conserveMemory', opts.conserveMemory, ...
        'sync', opts.sync);

    %%
    batch_time = toc(batch_time);
    speed = numel(batch)/batch_time;
    info.val = updateError(opts, info.val, net, res, batch_time);

    
    fprintf(' %.2f s (%.1f images/s)', batch_time, speed);
    n = t + numel(batch) - 1;
    switch opts.errorType,
    case 'multiclass'
      fprintf(' err %.3f cls_err %.3f', info.val.error(end)/n*100, info.val.cls_err(end)/n*100);
    case 'binary'
      fprintf(' err %.1f ap %.3f iou %.3f', info.val.error(end)/n*100, info.val.ap(end)/n*100, ...
        info.val.iou(end)/n*100);
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

