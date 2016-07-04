function res = matconvnet_simpledagnn(net, x, dzdy, res, varargin)
% Created by Changhan Wang, Jan. 16, 2015
% Modified by Xinchen Yan, Jan. 20, 2015
% Modified by Xinchen Yan, Feb. 16, 2015 (Issues in evaluation: 'sum')
% MATCONVNET_SIMPLEDAGNN  Evaluates a CNN with DAG topology
% (single source & single target)
% Some issues that might be incompatible with examples/cnn_train.m:
% 1) the layers can be organized at any order, the first hidden layer doesn't have to be net.layers{1}
% 2) the derivative of network output relative to parameters of layer i is stored in res(i+1).dzdw rather than res(i).dzdw

opts.res = [];
opts.conserveMemory = false;
opts.sync = false;
opts.disableDropout = false;
opts.freezeDropout = false;
opts = vl_argparse(opts, varargin);

n = numel(net.layers);

if (nargin <= 2) || isempty(dzdy)
    doder = false;
else
    doder = true;
end

gpuMode = isa(x, 'gpuArray');

if nargin <= 3 || isempty(res)
    res = struct(...
        'x', cell(1,n+1), ...
        'dzdx', cell(1,n+1), ...
        'dzdw', cell(1,n+1), ...
        'aux', cell(1,n+1), ...
        'time', num2cell(zeros(1,n+1)), ...
        'backwardTime', num2cell(zeros(1,n+1)));
end

% topological sorting
count = zeros(n, 1);
for i = 1:n, count(net.layers{i}.out) = count(net.layers{i}.out) + 1; end
queue = [];
pioneer = find(count == 0); %% TODO: multi-source
while ~isempty(pioneer) > 0
    queue = [queue, pioneer(1)];
    count(pioneer(1)) = -1;
    count(net.layers{pioneer(1)}.out) = count(net.layers{pioneer(1)}.out) - 1;
    pioneer = find(count == 0);
end

%queue
res(1).x = x ;
for i = 1:n
    cur = queue(i);
    l = net.layers{cur};
    if numel(l.in) == 1, pre = l.in; end
    if isempty(l.in), pre = 0; end
    res(pre+1).time = tic;
    switch l.type
        case 'conv'
            res(cur+1).x = vl_nnconv(res(pre+1).x, l.filters, l.biases,...
                'pad', l.pad, 'stride', l.stride);
        case 'pool'
            res(cur+1).x = vl_nnpool(res(pre+1).x, l.pool, 'pad', l.pad,...
                'stride', l.stride, 'method', l.method);
        case 'unpool'
            res(cur+1).x = vl_nnunpool(res(pre+1).x, 'stride', l.stride);
        case 'normalize'
            res(cur+1).x = vl_nnnormalize(res(pre+1).x, l.param);
        case 'softmax'
            res(cur+1).x = vl_nnsoftmax(res(pre+1).x);
        case 'loss'
            res(cur+1).x = vl_nnloss(res(pre+1).x, l.class);
        case 'softmaxloss'
            res(cur+1).x = vl_nnsoftmaxloss(res(pre+1).x, l.class);
        case 'relu'
            res(cur+1).x = vl_nnrelu(res(pre+1).x);
        case 'noffset'
            res(cur+1).x = vl_nnnoffset(res(pre+1).x, l.param);
        case 'dropout'
            if opts.disableDropout
                res(cur+1).x = res(pre+1).x;
            elseif opts.freezeDropout
                [res(cur+1).x, res(cur+1).aux] = vl_nndropout(res(pre+1).x,...
                    'rate', l.rate, 'mask', res(cur+1).aux);
            else
                [res(cur+1).x, res(cur+1).aux] = vl_nndropout(res(pre+1).x, 'rate', l.rate);
            end
        case 'sum'
            res(cur+1).x = res(l.in(1)+1).x;
            for j = 2:numel(l.in)
                res(cur+1).x = res(cur+1).x + res(l.in(j)+1).x;
            end
        case 'custom'
            res(cur+1) = l.forward(l, res(pre+1), res(cur+1));
        otherwise
            error('Unknown layer type %s', l.type);
    end
    
    % This should make things slower, but on MATLAB 2014a it is necessary
    % for any decent performance.
    if gpuMode && opts.sync, wait(gpuDevice); end
    res(pre+1).time = toc(res(pre+1).time);
end

if doder
    % initialization
    res(n+1).dzdx = dzdy;
    for i = n:-1:1
      if gpuMode,
        res(i).dzdx = gpuArray(zeros(size(gather(res(i).x)), 'single'));
      else
        res(i).dzdx = zeros(size(res(i).x), 'single');
      end
    end

    % backprop
    for i = n:-1:1
      cur = queue(i);
      indeg = numel(net.layers{cur}.in);
      if indeg == 0,
        indeg = 1;
      end

      for j = 1:indeg,
        if numel(net.layers{cur}.in) > 0
          pre = net.layers{cur}.in(j);
        else 
          pre = 0;
        end

        l = net.layers{cur};
        switch l.type
          case 'conv'
            %% TODO res(cur+1).dzdw{1} or res(pre+1).dzdw{1}
            [dzdx, res(cur+1).dzdw{1}, res(cur+1).dzdw{2}] = ...
              vl_nnconv(res(pre+1).x, l.filters, l.biases, res(cur+1).dzdx, ...
              'pad', l.pad, 'stride', l.stride);

          case 'pool'
            dzdx = vl_nnpool(res(pre+1).x, l.pool, res(cur+1).dzdx, ...
              'pad', l.pad, 'stride', l.stride, 'method', l.method);
          case 'unpool'
            dzdx = vl_nnunpool(res(pre+1).x, res(cur+1).dzdx, 'stride', l.stride);
          case 'normalize'
            dzdx = vl_nnnormalize(res(pre+1).x, l.param, res(cur+1).dzdx);
          case 'softmax'
            dzdx = vl_nnsoftmax(res(pre+1).x, res(cur+1).dzdx);
          case 'loss'
            dzdx = vl_nnloss(res(pre+1).x, l.class, res(cur+1).dzdx);
          case 'softmaxloss'
            dzdx = vl_nnsoftmaxloss(res(pre+1).x, l.class, res(cur+1).dzdx);
          case 'relu'
            dzdx = vl_nnrelu(res(pre+1).x, res(cur+1).dzdx);
          case 'noffset'
            dzdx = vl_nnnoffset(res(pre+1).x, l.param, res(cur+1).dzdx);
          case 'dropout'
            if opts.disableDropout
              dzdx = res(cur+1).dzdx;
            else
              dzdx = vl_nndropout(res(pre+1).x, res(cur+1).dzdx, 'mask', res(cur+1).aux);
            end
          case 'sum'
            dzdx = res(cur+1).dzdx;%% We should avoid the summation layer
          case 'custom'
            % TODO: finish this part
            res(pre+1) = l.backward(l, res(pre+1), res(cur+1));
        end
        res(pre+1).dzdx = res(pre+1).dzdx + dzdx;
      end

      if gpuMode && opts.sync, wait(gpuDevice); end

    end

end
