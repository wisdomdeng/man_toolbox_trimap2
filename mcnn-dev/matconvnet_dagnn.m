function res = matconvnet_dagnn(net, spec, din, dout, res, varargin)
% Created by Xinchen Yan, Feb. 15, 2015
% Modified by Xinchen Yan, March. 10, 2015 (fix some bugs and add sigmoid & tensorprod)
%
% MATCONVNET_DAGNN Evaluates a CNN with DAG topology
% (multi-source & multi-target)
% 
% Some issues that might be incompatible with examples/cnn_train.m & 'net' structure:
% 1) the layers can be organized at any order: 'name' is added to identify each layer;
% 2) the derivative of network output relative to parameters of layer i is aved in
% res(i+1).dzdw rather than res(i).dzdw
% 
% net.layers{:}.name: string
% spec.queue: vector 
% spec.inv_table: vector (name to index)
% din.layers{:}.x, din.layers{:}.name
% dout.layers{:}.dzdy, dout.layers{:}.name

opts.res = [];
opts.conserveMemory = false;
opts.sync = false;
opts.disableDropout = false;
opts.freezeDropout = false;
opts.trimap = [];
opts = vl_argparse(opts, varargin);

n = numel(net.layers);
nin = numel(din.layers);

%nout = numel(dout.layers);

%% setup
if (nargin <= 3) || isempty(dout)
  doder = false;
else
  doder = true;
end

gpuMode = isa(din.layers{1}.x, 'gpuArray');

if nargin <= 4 || isempty(res)
  res = struct(...
    'x', cell(1,n+nin), ...
    'dzdx', cell(1,n+nin), ...
    'dzdw', cell(1,n+nin), ...
    'aux', cell(1,n+nin), ...
    'time', num2cell(zeros(1,n+nin)), ...
    'backwardTime', num2cell(zeros(1,n+nin)));
end

%% queue
for i = 1:nin
  res(i).x = din.layers{i}.x;
end

for i = 1:n
  cur = spec.queue(i);
  l = net.layers{cur};
  if numel(l.in) == 1, pre = getIndx(l.in{1}, spec); end
  if isempty(l.in), pre = cur - nin; end
  res(pre+nin).time = tic;
  switch l.type
  case 'conv'
    res(cur+nin).x = vl_nnconv(res(pre+nin).x, l.filters, l.biases,...
      'pad', l.pad, 'stride', l.stride);
  case 'pool'
    res(cur+nin).x = vl_nnpool(res(pre+nin).x, l.pool, 'pad', l.pad,...
      'stride', l.stride, 'method', l.method);
  case 'unpool'
    res(cur+nin).x = vl_nnunpool(res(pre+nin).x, 'stride', l.stride);
  case 'normalize'
    res(cur+nin).x = vl_nnnormalize(res(pre+nin).x, l.param);
  case 'softmax'
    res(cur+nin).x = vl_nnsoftmax(res(pre+nin).x);
  case 'loss'
    res(cur+nin).x = vl_nnloss(res(pre+nin).x, l.class);
  case 'weightedloss'
    res(cur+nin).x = vl_nnweightedloss(res(pre+nin).x, l.class);
  case 'softmaxloss'
    res(cur+nin).x = vl_nnsoftmaxloss(res(pre+nin).x, l.class);
  case 'trimaploss'
    res(cur+nin).x = vl_nntrimaploss(res(pre+nin).x, l.class);
  case 'relu'
    res(cur+nin).x = vl_nnrelu(res(pre+nin).x);
  case 'noffset'
    res(cur+nin).x = vl_nnnoffset(res(pre+nin).x, l.param);
  case 'dropout'
    if opts.disableDropout
      res(cur+nin).x = res(pre+nin).x;
    elseif opts.freezeDropout
      [res(cur+nin).x, res(cur+nin).aux] = vl_nndropout(res(pre+nin).x,...
        'rate', l.rate, 'mask', res(i+nin).aux);
    else
      [res(cur+nin).x, res(cur+nin).aux] = vl_nndropout(res(pre+nin).x, 'rate', l.rate);
    end
  %% new layers added
  case 'tensorprod' %% only support 2-way input
    pre_u = getIndx(l.in{1}, spec);
    pre_v = getIndx(l.in{2}, spec);
    res(cur+nin).x = matconvnet_nntensorprod(res(pre_u+nin).x, res(pre_v+nin).x, ...
      l.filters, l.biases);
  case 'sigmoid'
    res(cur+nin).x = matconvnet_nnsigmoid(res(pre+nin).x);
  case 'concat'
    res(cur+nin).x = res(getIndx(l.in{1}, spec)+nin).x;
    for j = 2:numel(l.in)
      res(cur+nin).x = cat(3, res(cur+nin).x, res(getIndx(l.in{j}, spec)+nin).x);
    end
  case 'subset'
    res(cur+nin).x = res(pre+nin).x(:,:,l.ranges,:);
  case 'euloss'
    res(cur+nin).x = matconvnet_nneuloss(res(pre+nin).x, l.class); %% TODO
  case 'maskeuloss'
    res(cur+nin).x = matconvnet_nnmaskeuloss(res(pre+nin).x, l.class, l.mask); %% TODO
  case 'unpool'
    res(cur+nin).x = vl_nnunpool(res(pre+nin).x, 'stride', l.stride);
  case 'reshape'
    %size(gather(res(pre+nin).x))
    %l.winsize
    res(cur+nin).x = matconvnet_nnreshape(res(pre+nin).x, l.winsize);
  case 'custom'
    res(cur+nin) = l.forward(l, res(pre+nin), res(cur+nin));
  otherwise
    error('Unknown layer type %s', l.type);
  end
  % This should make things slower, but on MATLAB 2014a it is necessary
  % for any decent performance.
  if gpuMode && opts.sync, wait(gpuDevice); end
  res(pre+nin).time = toc(res(pre+nin).time);

  %[cur size(gather(res(cur+nin).x))]
end

%% backward
if doder

  nout = numel(dout.layers);

  for i = 1:nout
    res(n+nin-nout+i).dzdx = dout.layers{i}.dzdy;
  end

  for i = n:-1:1
    cur = spec.queue(i);
    l = net.layers{cur};
    if numel(l.in) == 1, pre = getIndx(l.in{1}, spec); end
    if isempty(l.in), pre = cur - nin; end
    
    switch l.type
    case 'conv'
      [res(pre+nin).dzdx, res(cur+nin).dzdw{1}, res(cur+nin).dzdw{2}] = ...
        vl_nnconv(res(pre+nin).x, l.filters, l.biases, res(cur+nin).dzdx, ...
        'pad', l.pad, 'stride', l.stride);
    case 'pool'
      res(pre+nin).dzdx = vl_nnpool(res(pre+nin).x, l.pool, res(cur+nin).dzdx, ...
        'pad', l.pad, 'stride', l.stride, 'method', l.method);
    case 'unpool'
      res(pre+nin).dzdx = vl_nnunpool(res(pre+nin).x, res(cur+nin).dzdx, 'stride', l.stride);
    case 'normalize'
      res(pre+nin).dzdx = vl_nnnormalize(res(pre+nin).x, l.param, res(cur+nin).dzdx);
    case 'softmax'
      res(pre+nin).dzdx = vl_nnsoftmax(res(pre+nin).x, res(cur+nin).dzdx);
    case 'loss'
      res(pre+nin).dzdx = vl_nnloss(res(pre+nin).x, l.class, res(cur+nin).dzdx);
    case 'weightedloss'
      if ~isfield(l, 'weights')
        % assigning weights based on the inverse of the number of ground truth label
        weights = ones(size(res(pre+nin).x, 3), 1);
        temp_class = gather(l.class);
        for i2 = 1:size(res(pre+nin).x, 3)
          weights(i2) = 1/(sum(temp_class(:) == i2)+1);
        end
      else
        weights = l.weights;
      end
      res(pre+nin).dzdx = vl_nnweightedloss(res(pre+nin).x, l.class, weights, res(cur+nin).dzdx);
    case 'trimaploss'
      if ~isfield(l, 'weights')
        % assigning weights based on the inverse of the number of ground truth label
        weights = ones(size(res(pre+nin).x, 3), 1);
        temp_class = gather(l.class);
        for i2 = 1:size(res(pre+nin).x, 3)
          weights(i2) = 1/(sum(temp_class(:) == i2)+1);
        end
      else
        weights = l.weights;
      end
      res(pre+nin).dzdx = vl_nntrimaploss(res(pre+nin).x, l.class, weights, opts.trimap, res(cur+nin).dzdx);
    case 'softmaxloss'
      res(pre+nin).dzdx = vl_nnsoftmaxloss(res(pre+nin).x, l.class, res(cur+nin).dzdx);
    case 'relu'
      res(pre+nin).dzdx = vl_nnrelu(res(pre+nin).x, res(cur+nin).dzdx);
    case 'noffset'
      res(pre+nin).dzdx = vl_nnnoffset(res(pre+nin).x, l.param, res(cur+nin).dzdx);
    case 'dropout'
      if opts.disableDropout
        res(pre+nin).dzdx = res(pre+nin).dzdx;
      else
        res(pre+nin).dzdx = vl_nndropout(res(pre+nin).x, res(cur+nin).dzdx, 'mask', res(cur+nin).aux);
      end
    % new layers added
    case 'tensorprod'
      pre_u = getIndx(l.in{1}, spec);
      pre_v = getIndx(l.in{2}, spec);
      [res(pre_u+nin).dzdx, res(pre_v+nin).dzdx, res(cur+nin).dzdw{1}, res(cur+nin).dzdw{2}] = ...
        matconvnet_nntensorprod(res(pre_u+nin).x, res(pre_v+nin).x, l.filters, l.biases, res(cur+nin).dzdx);
    case 'sigmoid'
      res(pre+nin).dzdx = matconvnet_nnsigmoid(res(pre+nin).x, res(cur+nin).dzdx);
    case 'concat'
      nchan_s = 1;
      for j = 1:numel(l.in)
        nchan_cur = size(gather(res(getIndx(l.in{j}, spec)+nin).x), 3);
        nchan_e = nchan_s + nchan_cur - 1;
        res(getIndx(l.in{j}, spec)+nin).dzdx = res(cur+nin).dzdx(:,:,nchan_s:nchan_e,:);
        nchan_s = nchan_s + nchan_cur;
      end
    case 'subset' %% TODO: overlap
      res(pre+nin).dzdx(:,:,l.ranges,:) = res(cur+nin).dzdx;
    case 'euloss'
      res(pre+nin).dzdx = matconvnet_nneuloss(res(pre+nin).x, l.class, res(cur+nin).dzdx);
    case 'maskeuloss'
      res(pre+nin).dzdx = matconvnet_nnmaskeuloss(res(pre+nin).x, l.class, l.mask, res(cur+nin).dzdx);
    case 'unpool'
      res(pre+nin).dzdx = vl_nnunpool(res(pre+nin).x, res(cur+nin).dzdx, 'stride', l.stride);
    case 'reshape'
      res(pre+nin).dzdx = matconvnet_nnreshape(res(pre+nin).x, l.winsize, res(cur+nin).dzdx);
    case 'custom'
      res(pres+nin) = l.backward(l, res(pre+nin), res(cur+nin));
    end

    if gpuMode && opts.sync, wait(gpuDevice); end
  end

end % end of if

end

%% 

function index = getIndx(name, spec)
  index = spec.inv_table(name);
end
