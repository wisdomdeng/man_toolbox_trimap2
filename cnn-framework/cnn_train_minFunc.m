function [net, info] = cnn_train_minFunc(opts, net, imdb, getBatch, updateError)

%% preprocess
% TODO
imdb.images.data(:,:,1:3,:) = bsxfun(@minus, imdb.images.data(:,:,1:3,:), mean(imdb.images.data(:,:,1:3,:), 4));

if ~exist(opts.expPath), mkdir(opts.expPath); end

opts.train = find(imdb.images.set == 1);
opts.val = find(imdb.images.set == 2); 

%% net init
if opts.useGpu
  net = vl_simplenn_move(net, 'gpu');
end

%% train and validation
info = struct;
info = cnn_info_init(opts, info, 0);

rng(0);

for epoch=1:opts.numEpochs
  
  % fast-forward to where we stopped
  
  modelPath = fullfile(opts.expPath, 'net-epoch-%d.mat');
  modelFigPath = fullfile(opts.expPath, 'net-train.pdf');
  %modelPath
  if opts.continue
    if exist(sprintf(modelPath, epoch),'file'), continue; end
    if epoch > 1
      fprintf('resuming by loading epoch %d\n', epoch-1);
      load(sprintf(modelPath, epoch-1), 'net', 'info');
    end
  end
  
  train = opts.train(randperm(numel(opts.train)));
  val = opts.val;

  info = cnn_info_init(opts, info, 1);

  %% mini-batch mode -- training
  for t = 1:opts.batchSize:numel(train)
    batch = train(t:min(t+opts.batchSize-1, numel(train)));
    batch_time = tic;
    fprintf('training: epoch %02d: processing batch %3d of %3d ...\n', epoch, ...
            fix(t/opts.batchSize)+1, ceil(numel(train)/opts.batchSize)) ;
    
    [im, labels] = getBatch(imdb, batch) ;
    if opts.useGpu,
      im = gpuArray(single(im));
    end

    %% minFunc/
    options.maxIter = 1;
    options.maxFunEvals = 10;
    options.displays = 'on';
    options.method = opts.solver;

    l2reg = opts.weightDecay;
    theta = matconvnet_roll_pars(net);
    opttheta = minFunc(@(p) matconvnet_cost_mini(p, opts, net, im, labels, l2reg), theta, options);
    net = matconvnet_unroll_pars(opttheta, net, opts.useGpu);

    net.layers{end}.class = labels;
    res = matconvnet_simplenn(net, im, [], [], ...
        'conserveMemory', opts.conserveMemory, ...
        'sync', opts.sync);
    %%
    batch_time = toc(batch_time);
    speed = numel(batch)/batch_time;
    info.train = updateError(opts, info.train, net, res, batch_time);
    
    clear res;
    if opts.verbose,
      fprintf(' %.2f s (%.1f images/s)', batch_time, speed);
      n = t + numel(batch) -1;
      if strcmp(opts.errorType, 'binary'),
        fprintf(' err %.1f', info.train.error(end)/n*100);
        %fprintf(' err %.1f ap %.3f iou %.3f', info.train.error(end)/n*100, ...
        %  info.train.ap(end)/n*100, info.train.iou(end)/n*100);    
      elseif strcmp(opts.errorType, 'multiclass'),
        fprintf(' err %.1f err (top 5) %.1f', info.train.error(end)/n*100, ...
          info.train.topFiveError(end)/n*100);
      end
      fprintf('\n');
    end

  end 
  
  %% mini-batch mode -- validation
  for t = 1:opts.batchSize:numel(val)
    batch = val(t:min(t+opts.batchSize-1, numel(val)));
    batch_time = tic;
    fprintf('validation: epoch %02d: processing batch %3d of %3d ...\n', epoch, ...
       fix(t/opts.batchSize)+1, ceil(numel(val)/opts.batchSize)) ;
    [im, labels] = getBatch(imdb, batch);

    if opts.useGpu
      im = gpuArray(im);
    end
    
    net.layers{end}.class = labels;
    res = matconvnet_simplenn(net, im, [], [], ...
        'conserveMemory', opts.conserveMemory, ...
        'sync', opts.sync);

    %%
    batch_time = toc(batch_time);
    speed = numel(batch)/batch_time;
    info.val = updateError(opts, info.val, net, res, batch_time);

    clear res;
    if opts.verbose,
      fprintf(' %.2f s (%.1f images/s)', batch_time, speed);
      n = t + numel(batch) - 1;
      if strcmp(opts.errorType, 'binary'),
        fprintf(' err %.1f', info.val.error(end)/n*100);
        %fprintf(' err %.1f ap %.3f iou %.3f', info.val.error(end)/n*100, ...
        %  info.val.ap(end)/n*100, info.val.iou(end)/n*100);
      elseif strcmp(opts.errorType, 'multiclass'),
        fprintf(' err %.1f err (top 5) %.1f', info.val.error(end)/n*100, ...
          info.val.topFiveError(end)/n*100);
      end
      fprintf('\n');
    end

  end

  info.numtrain = numel(train);
  info.numval = numel(val);
  [info, opts] = cnn_drawcurve(info, opts, epoch);
  save(sprintf(modelPath, epoch), 'net', 'info');
  if opts.verbose,
     drawnow;
     print(1, modelFigPath, '-dpdf');
  end
 
end % end of for

modelPath = fullfile(opts.expPath, 'net-epoch-%d.mat');
load(sprintf(modelPath, opts.numEpochs), 'net', 'info');

end

