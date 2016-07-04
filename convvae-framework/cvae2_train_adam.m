function [net_enc, net_dec, net_style1, net_style2, info] = ...
  cvae2_train_adam(opts, net_enc, net_dec, net_style1, net_style2, ...
  imdb, getBatch, updateError)

nin = numel(imdb.din.layers);

if ~exist(opts.expDir), mkdir(opts.expDir); end

% train & validation
opts.train = find(imdb.set == 1);
opts.val = find(imdb.set == 2); 

info = struct;
info = vae_info_init(opts, info, 0);

%% net initialization
rng(0);
net_enc = vae_param_init(net_enc, opts, 'init');
net_dec = vae_param_init(net_dec, opts, 'init');
net_style1 = vae_param_init(net_style1, opts, 'init');
net_style2 = vae_param_init(net_style2, opts, 'init');

%% adam param initialization
lr = 0;
beta1 = opts.adam.beta1;
beta2 = opts.adam.beta2;
epsilon = opts.adam.epsilon;
%lambda = opts.adam.lambda;

res_enc = [];
res_dec = [];
res_style1 = [];
res_style2 = [];
one = cpu2gpu_copy(1.0, opts.useGPU);

numbatch = min(floor(min(numel(opts.train), 100000)/opts.batchSize), 20);
nround = ceil(opts.adam.decayround/numbatch);
%nround = ceil(10000/numbatch);
ii = 0;

%% fast-forward to where we stopped
if ~opts.continue
  epoch_start = 0;
else
  epoch_start = 0;
  while exist(sprintf('%s/net-epoch-%d.mat',opts.expPath, epoch_start+opts.saveround), 'file')
    epoch_start = epoch_start + opts.saveround;
  end
  if epoch_start > 0
    load(sprintf('%s/net-epoch-%d.mat', opts.expPath, epoch_start));
    fprintf('resuming by loading epoch %d\n', epoch_start);
  end
  %% initialize net, info, ii, lr
  for epoch = 1:epoch_start
    prevLr = lr ;
    lr = opts.adam.learningRate(min(epoch, numel(opts.adam.learningRate))) ;
    if epoch > nround,% TODO
      lr = lr /(1 +  opts.adam.lr_decay * (epoch-nround));
    end
    ii = ii + numel(1:opts.batchSize:numel(opts.train));
  end
end

for epoch=epoch_start+1:opts.numEpochs
  prevLr = lr ;
  lr = opts.adam.learningRate(min(epoch, numel(opts.adam.learningRate))) ;
  if epoch > nround,% TODO
    lr = lr / (1 + opts.adam.lr_decay * (epoch-nround));
  end

  modelPath = fullfile(opts.expPath, 'net-epoch-%d.mat');
  modelFigPath = fullfile(opts.expPath, 'net-train.pdf');
  
  
  %rng(epoch);
  train = opts.train(randperm(numel(opts.train)));
  val = opts.val;

  info = vae_info_init(opts, info, 1);
  %% mini-batch mode -- training
  for t = 1:opts.batchSize:numel(train)
    batch = train(t:min(t+opts.batchSize-1, numel(train)));
    batch_time = tic;
   
    fprintf('training: epoch %02d: processing batch %3d of %3d (lr = %g)...\n', epoch, ...
            fix(t/opts.batchSize)+1, ceil(numel(train)/opts.batchSize), lr) ;
    
    [data, style, noise] = getBatch(imdb, batch, opts) ;
    [cost, res_enc, res_dec, res_style1, res_style2] = ...
	cvae2_cost(data, style, noise, ...
        net_enc, net_dec, net_style1, net_style2, ...
        res_enc, res_dec, res_style1, res_style2);
    ii = ii + 1;
    % gradient update
    net_enc = vae_param_update_adam(opts, net_enc, res_enc, lr, beta1, beta2, epsilon, ii, 0);
    net_dec = vae_param_update_adam(opts, net_dec, res_dec, lr, beta1, beta2, epsilon, ii, 1);
    net_style1 = vae_param_update_adam(opts, net_style1, res_style1, lr, beta1, beta2, epsilon, ii, 0);
    net_style2 = vae_param_update_adam(opts, net_style2, res_style2, lr, beta1, beta2, epsilon, ii, 0);

    KL = gather(res_enc(end).x)  / opts.batchSize;
    LL = gather(res_dec(end).x)  / opts.batchSize;

    batch_time = toc(batch_time);
    speed = numel(batch)/batch_time;

    info.train = updateError(opts, info.train, KL, LL, batch_time);

    fprintf(' %.2f s (%.1f images/s)', batch_time, speed);
    n = t + numel(batch) -1;
    fprintf(' LB %.3f KL %.3f LL %.3f ', info.train.objective(end)/n, ...
          info.train.kl(end)/n, info.train.ll(end)/n);

    fprintf('\n');

  end 
  
  %% mini-batch mode -- validation
  for t = 1:opts.batchSize:numel(val)
    
    batch = val(t:min(t+opts.batchSize-1, numel(val)));
    batch_time = tic;
    fprintf('validation: epoch %02d: processing batch %3d of %3d ...\n', epoch, ...
       fix(t/opts.batchSize)+1, ceil(numel(val)/opts.batchSize)) ;

    [data, style, noise] = getBatch(imdb, batch, opts) ;
    [cost, res_enc, res_dec, res_style1, res_style2] = ...
      cvae2_cost(data, style, noise, ...
      net_enc, net_dec, net_style1, net_style2, ...
      res_enc, res_dec, res_style1, net_style2);

    KL = gather(res_enc(end).x)  / opts.batchSize;
    LL = gather(res_dec(end).x) / opts.batchSize;

    batch_time = toc(batch_time);
    speed = numel(batch)/batch_time;

    info.val = updateError(opts, info.val, KL, LL, batch_time);

    fprintf(' %.2f s (%.1f images/s)', batch_time, speed);
    n = t + numel(batch) - 1;
      
    fprintf(' LB %.3f KL %.3f LL %.3f ', info.val.objective(end)/n, ...
      info.val.kl(end)/n, info.val.ll(end)/n);

    fprintf('\n');
  
  end
  
  info.numtrain = numel(train);
  info.numval = numel(val);
  
  [info, opts] = vae_drawcurve(info, opts, epoch);
  
  %% save memory
  if mod(epoch, opts.saveround) == 0
    save(sprintf(modelPath, epoch), 'net_enc', 'net_dec', 'net_style1', 'net_style2', 'info');
  end

  if opts.verbose,
    drawnow;
    print(1, modelFigPath, '-dpdf');
  end

end % end of for

modelPath = fullfile(opts.expPath, 'net-epoch-%d.mat');
load(sprintf(modelPath, opts.numEpochs), 'net_enc', 'net_dec', 'net_style1','net_style2','info');

end

