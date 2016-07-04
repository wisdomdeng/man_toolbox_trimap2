%
% Modified by Xinchen Yan, March. 24, 2015 (fix matconvnet_dagnn_move & reset_params issue)

function [net, info] = dagcnn_train_adam1(opts, net, spec, imdb, getBatch, updateError)

%imdb.din.layers{:}.data
%imdb.dout.layers{:}.data
%imdb.set

nl = numel(net.layers);
if size(imdb, 2) == 1
    nin = numel(imdb.din.layers);
    nout = numel(imdb.dout.layers);
else
    % assuming every imdb has the same number of layers
    nin = numel(imdb(1).din.layers);
    nout = numel(imdb(1).dout.layers);
end
if ~exist(opts.expDir), mkdir(opts.expDir); end


if size(imdb, 2) == 1
    opts.train = find(imdb.set == 1);
    opts.val = find(imdb.set == 2); 
else
    numtrain = 0;
    numval = 0;
    for i = [1:size(imdb, 2)]
        numtrain = numtrain + imdb(i).numtrain;
        numval = numval + imdb(i).numval;
    end
    set = [ones(1, numtrain), 2 * ones(1, numval)];
    opts.train = find(set == 1);
    opts.val = find(set == 2); 
end
beta1 = opts.adam.beta1;
beta2 = opts.adam.beta2;
epsilon = opts.adam.epsilon;
numbatch = numel(1:opts.batchSize:numel(opts.train));
nround = ceil(opts.adam.decayround/numbatch);
ii = 0;
%% net init
for i = 1:nl
  switch net.layers{i}.type,
  case {'conv','tensorprod'}
    net.layers{i}.mfilters = zeros('like', net.layers{i}.filters) ;
    net.layers{i}.mbiases = zeros('like', net.layers{i}.biases) ;
    net.layers{i}.vfilters = zeros('like', net.layers{i}.filters) ;
    net.layers{i}.vbiases = zeros('like', net.layers{i}.biases) ;
  end
end



if opts.useGPU
  net = matconvnet_dagnn_move(net, 'gpu');
  for l = 1:nl;
    switch net.layers{l}.type
        case {'conv','tensorprod'}
            net.layers{l}.mfilters = gpuArray(net.layers{l}.mfilters);
            net.layers{l}.mbiases = gpuArray(net.layers{l}.mbiases);
            net.layers{l}.vfilters = gpuArray(net.layers{l}.vfilters);
            net.layers{l}.vbiases = gpuArray(net.layers{l}.vbiases);
    end
  end
end

%% train and validation
info = struct;
info = dagcnn_info_init(opts, info, 0);

rng(0);

lr = 0;
res = [];
one = cpu2gpu_copy(1.0, opts.useGPU);

% subject to change
if isfield(opts, 'trimapDir')
    load(fullfile(opts.trimapDir, 'trainTri_origin.mat'));
    imdb(1).din.layers{1}.trimap = trainTrimap;
    load(fullfile(opts.trimapDir, 'trainTri_NYU_origin.mat'));
    imdb(2).din.layers{1}.trimap = trainTrimap;
end

for epoch=1:opts.numEpochs
  prevLr = lr ;
  lr = opts.adam.learningRate(min(epoch, numel(opts.adam.learningRate))) ;
  mmt = opts.adam.momentum(min(epoch, numel(opts.adam.momentum)));
  if epoch > nround, 
    lr = lr / (1 + opts.adam.lr_decay * (epoch - nround));
  end

  % fast-forward to where we stopped
  modelPath = fullfile(opts.expPath, 'net-epoch-%d.mat');
  modelFigPath = fullfile(opts.expPath, 'net-train.pdf');
  %modelPath
  if opts.continue
    if exist(sprintf(modelPath, epoch),'file')
      ii = ii + numel(1:opts.batchSize:numel(opts.train));
      continue;
    end
    if epoch > 1
      fprintf('resuming by loading epoch %d\n', epoch-1);
      load(sprintf(modelPath, epoch-1), 'net', 'info');
      
    end
  end
  
  %todo::generate the trimap file for every epoch and reassign the the imdbtrimap 
  train = opts.train(randperm(numel(opts.train)));
  val = opts.val;

  info = dagcnn_info_init(opts, info, 1);
  % reset momentum if needed
  %if prevLr ~= lr
  %  fprintf('learning rate changed (%f --> %f): resetting momentum\n', prevLr, lr);
  %  for l = 1:nl,
  %    % TODO: matconvnet_dagnn_reset_params
  %    switch net.layers{l}.type
  %    case {'conv','tensorprod'}
  %      net.layers{l}.filtersMomentum = 0 * net.layers{l}.filtersMomentum;
  %      net.layers{l}.biasesMomentum = 0 * net.layers{l}.biasesMomentum;
  %    end
  %  end
  %end
  
  
  if isfield(opts, 'trimapDir') && mod(epoch, 10) == 1
    %rerun the trimap generation code
    command = ['python ', opts.trimapDir, 'TriMapsForCNNTrainingOnlyDependingOnGroundTruth.py ', opts.trimapDir, 'trainData.mat ', opts.trimapDir, 'trainTri.mat '];
    [status, cdmout] = system(command);
    if isequal(cdmout(1), '1')
        load(fullfile(opts.trimapDir, 'trainTri.mat'));
        imdb(1).din.layers{1}.trimap(1:imdb(1).numtrain) = trainTrimap(1:imdb(1).numtrain);
    else
        % it fails sometimes
        load(fullfile(opts.trimapDir, 'trainTri_origin.mat'));
        imdb(1).din.layers{1}.trimap = trainTrimap;
    end
    command = ['python ', opts.trimapDir, 'TriMapsForCNNTrainingOnlyDependingOnGroundTruth.py ', opts.trimapDir, 'trainData_NYU.mat ', opts.trimapDir, 'trainTri_NYU.mat '];
    [status, cdmout] = system(command);
    if isequal(cdmout(1), '1')
        load(fullfile(opts.trimapDir, 'trainTri_NYU.mat'));
        imdb(2).din.layers{1}.trimap(1:imdb(2).numtrain) = trainTrimap(1:imdb(2).numtrain);
    else
        load(fullfile(opts.trimapDir, 'trainTri_NYU_origin.mat'));
        imdb(2).din.layers{1}.trimap = trainTrimap;
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
      if isfield(din.layers{i}, 'trimap')
          din.layers{i}.trimap = cpu2gpu_copy(din.layers{i}.trimap, opts.useGPU);
      end
    end
    for i = 1:nout,
      dout.layers{i}.x = cpu2gpu_copy(dout.layers{i}.x, opts.useGPU);
      if isfield(dout.layers{i},'mask'),
        dout.layers{i}.mask = cpu2gpu_copy(dout.layers{i}.mask, opts.useGPU);
      end
      dout.layers{i}.dzdy = cpu2gpu_copy(dout.layers{i}.dzdy, opts.useGPU);
    end

    %% TODO: clean up cpu2gpu_copy

    % Backprop Update
    for i = 1:nout,
      net.layers{nl-nout+i}.class = cpu2gpu_copy(dout.layers{i}.x, opts.useGPU);
      if isfield(dout.layers{i},'mask'),
        net.layers{nl-nout+i}.mask = cpu2gpu_copy(dout.layers{i}.mask, opts.useGPU);
      end
    end

    %%net_update = net;
    %temporary solution is to assume the trimap is in the first layer
    if isfield(opts, 'trimapDir')
        trimap = 1 + sum(din.layers{1}.trimap(:, :, 1:2, :), 3);
        % nestrov momentum
        %%if opts.sgd.nestrov
        %%  for l = 1:nl
        %%    switch net_update.layers{l}.type
        %%    case {'conv','tensorprod'}
        %%      net_update.layers{l}.filters = net_update.layers{l}.filters + ...
        %%        mmt * net_update.layers{l}.filtersMomentum;
        %%      net_update.layers{l}.biases = net_update.layers{l}.biases + ...
        %%        mmt * net_update.layers{l}.biasesMomentum;
        %%    end
        %%  end
        %%end

        %% TODO: fill in l.class
    
        res = matconvnet_dagnn(net, spec, din, dout, res, ...
            'conserveMemory', opts.conserveMemory, ...
            'sync', opts.sync,...
            'trimap', trimap);
    else
        res = matconvnet_dagnn(net, spec, din, dout, res, ...
            'conserveMemory', opts.conserveMemory, ...
            'sync', opts.sync);
    end
    ii = ii + 1; 
    %clear net_update;
    % gradient step
    %for l = 1:nl
    %  switch net.layers{l}.type
    %  case {'conv','tensorprod'}
    %    net.layers{l}.filtersMomentum = ...
    %      mmt * net.layers{l}.filtersMomentum ...
    %      - (lr * net.layers{l}.filtersLearningRate) * ...
    %      (opts.weightDecay * net.layers{l}.filtersWeightDecay) * net.layers{l}.filters ...
    %      - (lr * net.layers{l}.filtersLearningRate) / numel(batch) * res(l+nin).dzdw{1};
    %  
    %    net.layers{l}.biasesMomentum = ...
    %    mmt * net.layers{l}.biasesMomentum ...
    %      - (lr * net.layers{l}.biasesLearningRate) * ...
    %      (opts.weightDecay * net.layers{l}.biasesWeightDecay) * net.layers{l}.biases ...
    %      - (lr * net.layers{l}.biasesLearningRate) / numel(batch) * res(l+nin).dzdw{2};
    %
    %    net.layers{l}.filters = net.layers{l}.filters + net.layers{l}.filtersMomentum;
    %    net.layers{l}.biases = net.layers{l}.biases + net.layers{l}.biasesMomentum;
    %  end
    %
    %end

    %if opts.sgd.nestrov
    %  res = matconvnet_dagnn(net, spec, din, [], [], ...
    %    'conserveMemory', opts.conserveMemory, ...
    %    'sync', opts.sync);
    %end

    lr_forupdate = lr;
    if ii < opts.adam.decayround,
      lr_forupdate = lr * sqrt(1 - (1 - beta2).^ii) ./ (1 - (1 - beta1).^ii);
    end

    %% TODO: Finish the parameter update parameter
    for l = 1:nl
      switch net.layers{l}.type
      case {'conv','tensorprod'}
          %gram_parameter
          grad_filters = res(l+nin).dzdw{1} - net.layers{l}.filters * opts.weightDecay;
          grad_biases = res(l+nin).dzdw{2} - net.layers{l}.biases * opts.weightDecay;
          %mgrad
          net.layers{l}.mfilters = beta1 * net.layers{l}.mfilters + (1 - beta1) * grad_filters;
          net.layers{l}.mbiases = beta1 * net.layers{l}.mbiases + (1 - beta1) * grad_biases;
          %vgrad
          net.layers{l}.vfilters = beta2 * net.layers{l}.vfilters + (1 - beta2) * grad_filters.^2;
          net.layers{l}.vbiases = beta2 * net.layers{l}.vbiases + (1 - beta2) * grad_biases.^2;
          %delta
          delta_filters = net.layers{l}.mfilters ./ (sqrt(net.layers{l}.vfilters) + epsilon);
          delta_biases = net.layers{l}.mbiases ./ (sqrt(net.layers{l}.vbiases) + epsilon);
          %update
          net.layers{l}.filters = net.layers{l}.filters - cpu2gpu_copy(lr * delta_filters, opts.useGPU);
          net.layers{l}.biases = net.layers{l}.biases - cpu2gpu_copy(lr * delta_biases, opts.useGPU);          
      end
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
    %  dout.layers{i}.dzdy = cpu2gpu_copyf(dout.layers{i}.dzdy, opts.useGPU);
    %end
    
    %% forward
    for i = 1:nout,
      net.layers{nl-nout+i}.class = cpu2gpu_copy(dout.layers{i}.x, opts.useGPU);
      if isfield(dout.layers{i},'mask'),
        net.layers{nl-nout+i}.mask = cpu2gpu_copy(dout.layers{i}.mask, opts.useGPU);
      end
    end
    if isfield(opts, 'trimapDir')
        trimap = 1 + sum(din.layers{1}.trimap(:, :, 1:2, :));
        res = matconvnet_dagnn(net, spec, din, [], [], ...
            'conserveMemory', opts.conserveMemory, ...
            'sync', opts.sync,...
            'trimap', trimap);
    else
        res = matconvnet_dagnn(net, spec, din, [], [], ...
            'conserveMemory', opts.conserveMemory, ...
            'sync', opts.sync);
    end
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

