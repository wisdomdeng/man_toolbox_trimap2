%
% Modified by Xinchen Yan, March. 24, 2015 (fix matconvnet_dagnn_move & reset_params issue)

function [net, info] = dagcnn_train_sgd(opts, net, spec, imdb, getBatch, updateError)

%imdb.din.layers{:}.data
%imdb.dout.layers{:}.data
%imdb.set

nl = numel(net.layers);
nin = numel(imdb.din.layers);
nout = numel(imdb.dout.layers);

if ~exist(opts.expDir), mkdir(opts.expDir); end

opts.train = find(imdb.set == 1);
opts.val = find(imdb.set == 2); 

%% net init
for i = 1:nl
  switch net.layers{i}.type,
  case {'conv','tensorprod'}
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
end

if opts.useGPU
  net = matconvnet_dagnn_move(net, 'gpu');
  %for i = 1:nl,
  %  if ~strcmp(net.layers{i}.type,'conv'), continue; end
  %  net.layers{i}.filtersMomentum = gpuArray(net.layers{i}.filtersMomentum) ;
  %  net.layers{i}.biasesMomentum = gpuArray(net.layers{i}.biasesMomentum) ;
  %end
end

%% train and validation
info = struct;
info = dagcnn_info_init(opts, info, 0);

rng(0);

lr = 0;
res = [];
one = cpu2gpu_copy(1.0, opts.useGPU);

for epoch=1:opts.numEpochs
  prevLr = lr ;
  lr = opts.sgd.learningRate(min(epoch, numel(opts.sgd.learningRate))) ;
  mmt = opts.sgd.momentum(min(epoch, numel(opts.sgd.momentum)));

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

  info = dagcnn_info_init(opts, info, 1);
  % reset momentum if needed
  if prevLr ~= lr
    fprintf('learning rate changed (%f --> %f): resetting momentum\n', prevLr, lr);
    for l = 1:nl,
      % TODO: matconvnet_dagnn_reset_params
      switch net.layers{l}.type
      case {'conv','tensorprod'}
        net.layers{l}.filtersMomentum = 0 * net.layers{l}.filtersMomentum;
        net.layers{l}.biasesMomentum = 0 * net.layers{l}.biasesMomentum;
      end
    end
  end

  %% mini-batch mode -- training
  for t = 1:opts.batchSize:numel(train)
    batch = train(t:min(t+opts.batchSize-1, numel(train)));
    batch_time = tic;
    fprintf('training: epoch %02d: processing batch %3d of %3d (lr = %g, mmt = %g)...\n', epoch, ...
            fix(t/opts.batchSize)+1, ceil(numel(train)/opts.batchSize), lr, mmt) ;
    
    [din, dout] = getBatch(imdb, batch) ;
    for i = 1:nin,
      din.layers{i}.x = cpu2gpu_copy(din.layers{i}.x, opts.useGPU);
    end
    for i = 1:nout,
      dout.layers{i}.x = cpu2gpu_copy(dout.layers{i}.x, opts.useGPU);
      if isfield(dout.layers{i},'mask'),
        dout.layers{i}.mask = cpu2gpu_copy(dout.layers{i}.mask, opts.useGPU);
      end
      dout.layers{i}.dzdy = cpu2gpu_copy(dout.layers{i}.dzdy, opts.useGPU);
    end

    %% TODO: clean up cpu2gpu_copy

    %% Backprop Update
    for i = 1:nout,
      net.layers{nl-nout+i}.class = cpu2gpu_copy(dout.layers{i}.x, opts.useGPU);
      if isfield(dout.layers{i},'mask'),
        net.layers{nl-nout+i}.mask = cpu2gpu_copy(dout.layers{i}.mask, opts.useGPU);
      end
    end

    net_update = net;
    
    % nestrov momentum
    if opts.sgd.nestrov
      for l = 1:nl
        switch net_update.layers{l}.type
        case {'conv','tensorprod'}
          net_update.layers{l}.filters = net_update.layers{l}.filters + ...
            mmt * net_update.layers{l}.filtersMomentum;
          net_update.layers{l}.biases = net_update.layers{l}.biases + ...
            mmt * net_update.layers{l}.biasesMomentum;
        end
      end
    end

    %% TODO: fill in l.class
    res = matconvnet_dagnn(net_update, spec, din, dout, res, ...
        'conserveMemory', opts.conserveMemory, ...
        'sync', opts.sync);
    
    clear net_update;
    % gradient step
    for l = 1:nl
      switch net.layers{l}.type
      case {'conv','tensorprod'}
        net.layers{l}.filtersMomentum = ...
          mmt * net.layers{l}.filtersMomentum ...
          - (lr * net.layers{l}.filtersLearningRate) * ...
          (opts.weightDecay * net.layers{l}.filtersWeightDecay) * net.layers{l}.filters ...
          - (lr * net.layers{l}.filtersLearningRate) / numel(batch) * res(l+nin).dzdw{1};
      
        net.layers{l}.biasesMomentum = ...
        mmt * net.layers{l}.biasesMomentum ...
          - (lr * net.layers{l}.biasesLearningRate) * ...
          (opts.weightDecay * net.layers{l}.biasesWeightDecay) * net.layers{l}.biases ...
          - (lr * net.layers{l}.biasesLearningRate) / numel(batch) * res(l+nin).dzdw{2};

        net.layers{l}.filters = net.layers{l}.filters + net.layers{l}.filtersMomentum;
        net.layers{l}.biases = net.layers{l}.biases + net.layers{l}.biasesMomentum;
      end

    end

    if opts.sgd.nestrov
      res = matconvnet_dagnn(net, spec, din, [], [], ...
        'conserveMemory', opts.conserveMemory, ...
        'sync', opts.sync);
    end

    batch_time = toc(batch_time);
    speed = numel(batch)/batch_time;
    info.train = updateError(opts, info.train, net, res, batch_time);
    
    if opts.verbose,
      fprintf(' %.2f s (%.1f images/s)', batch_time, speed);
      n = t + numel(batch) -1;
      
      switch opts.errorType
      case 'binary'
        fprintf(' err %.1f ap %.3f iou %.4f', info.train.error(end)/n*100, info.train.ap(end)/n*100, ...
          info.train.iou(end)/n*100);
      case 'multiclass'
        fprintf(' err %.3f cls_err %.3f', info.train.error(end)/n*100, info.train.class_error(end)/n*100);
      case 'recon'
        fprintf(' err %.3f sat %.3f sat2 %.3f', info.train.error(end)/n, ...
          info.train.sat(end)/n*100, info.train.sat2(end)/n*100);
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

    [din, dout] = getBatch(imdb, batch);
    for i = 1:nin,
      din.layers{i}.x = cpu2gpu_copy(din.layers{i}.x, opts.useGPU);
    end
    %for i = 1:nout,
    %  dout.layers{i}.x = cpu2gpu_copy(dout.layers{i}.x, opts.useGPU);
    %  dout.layers{i}.dzdy = cpu2gpu_copy(dout.layers{i}.dzdy, opts.useGPU);
    %end
    
    %% forward
    for i = 1:nout,
      net.layers{nl-nout+i}.class = cpu2gpu_copy(dout.layers{i}.x, opts.useGPU);
      if isfield(dout.layers{i},'mask'),
        net.layers{nl-nout+i}.mask = cpu2gpu_copy(dout.layers{i}.mask, opts.useGPU);
      end
    end

    res = matconvnet_dagnn(net, spec, din, [], [], ...
        'conserveMemory', opts.conserveMemory, ...
        'sync', opts.sync);

    %%
    batch_time = toc(batch_time);
    speed = numel(batch)/batch_time;

    %TODO
    info.val = updateError(opts, info.val, net, res, batch_time);

    if opts.verbose,
      fprintf(' %.2f s (%.1f images/s)', batch_time, speed);
      n = t + numel(batch) - 1;
      switch opts.errorType
      case 'binary'
        fprintf(' err %.1f ap %.3f iou %.3f', info.val.error(end)/n*100, info.val.ap(end)/n*100, ...
          info.val.iou(end)/n*100);
      case 'multiclass'
        fprintf(' err %.3f cls_err %.3f', info.val.error(end)/n*100, info.val.class_error(end)/n*100);
      case 'recon'
        fprintf(' err %.3f sat %.3f sat2 %.3f', info.val.error(end)/n, ...
          info.val.sat(end)/n*100, info.val.sat2(end)/n*100);
      end
      fprintf('\n');
    end

  end
  
  info.numtrain = numel(train);
  info.numval = numel(val);
  [info, opts] = dagcnn_drawcurve(info, opts, epoch);
  save(sprintf(modelPath, epoch), 'net', 'info');
  if opts.verbose,
    drawnow;
    print(1, modelFigPath, '-dpdf');
  end

end % end of for

modelPath = fullfile(opts.expPath, 'net-epoch-%d.mat');
load(sprintf(modelPath, opts.numEpochs), 'net', 'info');

end

