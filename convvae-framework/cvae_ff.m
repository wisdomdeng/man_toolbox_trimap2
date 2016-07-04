function res = cvae_ff(net, x, s, style, noise, res, varargin)


% -------------------------------------------------------------------------
%                                                      default parameters
% -------------------------------------------------------------------------

opts.res = [];
opts.conserveMemory = false;
opts.sync = false;
opts.disableDropout = false;
opts.freezeDropout = false;
opts.gradcheck = false;

opts = vl_argparse(opts, varargin);


% -------------------------------------------------------------------------
%                                                        number of layers
% -------------------------------------------------------------------------

n = numel(net.layers);


% -------------------------------------------------------------------------
%                                                                     gpu
% -------------------------------------------------------------------------

gpuMode = isa(x, 'gpuArray');
one = single(1);
zero = single(0);
if gpuMode,
    one = gpuArray(one);
    zero = gpuArray(zero);
end

if ~exist('res', 'var') || isempty(res)
    res = struct(...
        'x', cell(1,n+1), ...
        's', cell(1,n+1), ...
        'dzdx', cell(1,n+1), ...
        'dzds', cell(1,n+1), ...
        'filters', cell(1,n+1), ...
        'biases', cell(1,n+1), ...
        'filters_std', cell(1,n+1), ...
        'biases_std', cell(1,n+1), ...
        'aux', cell(1,n+1), ...
        'time', num2cell(zeros(1,n+1)), ...
        'backwardTime', num2cell(zeros(1,n+1)));
end

if isempty(s),
    res(1).x = x;
else
    res(1).x = x;
    res(1).s = s;
end

if gpuMode,
    one = gpuArray(one);
    zero = gpuArray(zero);
end


% -------------------------------------------------------------------------
%                                                     forward propagation
% 1. linear response    : conv
% 2. sampling           : gaussian_sampling
% 3. pooling            : pool, unpool
% 4. nonlinearities     : relu, sigmoid, softplus, softmax, normalize
% 5. loss               : bernoulli_loss, gaussian_loss, softmax_loss,
%                         kldiv_loss (to be minimized)
% 6. dropout            : dropout
% 7. fc_mix             : concatenation+fc layer (z + style)
% -------------------------------------------------------------------------

for i = 1:n,
    l = net.layers{i};
    res(i).time = tic;
    
    switch l.type
        case 'conv',
            % linear (convolutional) response
            if opts.gradcheck,
                for nh = 1:size(l.filters, 4),
                    res(i+1).x(:, :, nh, :) = convn(res(i).x, l.filters(end:-1:1, end:-1:1, end:-1:1, nh), 'valid') + l.biases(nh);
                end
            else
                res(i+1).x = vl_nnconv(res(i).x, l.filters, l.biases, 'pad', l.pad, 'stride', l.stride);
            end
        case 'conv_part'
            res(i+1).x = vl_nnconv_part_ff(res(i).x, l.filters, l.biases, ...
              'pad', l.pad, 'stride', l.stride, 'group', l.group);
          
        case 'conv_gaussian',
            if ~isfield(l, 'share_std'),
              l.share_std = 0;
            end

            if l.share_std == 1, % spatial only
              l.filters_std = repmat(l.filters_std, [size(l.filters, 1), size(l.filters, 2), 1, 1]);
            elseif l.share_std == 2, %% spatial + channel
              l.filters_std = repmat(l.filters_std, [size(l.filters, 1), size(l.filters, 2), 1, size(l.filters, 4)]);
              l.biases_std = repmat(l.biases_std, [1, size(l.filters, 4)]);
            end
            % linear response (mean)
            res(i+1).x = vl_nnconv(res(i).x, l.filters, l.biases, 'pad', l.pad, 'stride', l.stride);
           
            % linear response (std)
            res(i+1).s = vl_nnconv(res(i).x, l.filters_std, l.biases_std, 'pad', l.pad, 'stride', l.stride);
            res(i+1).s = exp(0.5*res(i+1).s);
        
        case 'fc_mix_gaussian',
            if ~isfield(l, 'share_std'),
              l.share_std = 0;
            end

            if l.share_std == 1, % spatial only
              l.filters_std = repmat(l.filters_std, [size(l.filters, 1), size(l.filters, 2), 1, 1]);
            elseif l.share_std == 2, %% spatial + channel
              l.filters_std = repmat(l.filters_std, [size(l.filters, 1), size(l.filters, 2), 1, size(l.filters, 4)]);
              l.biases_std = repmat(l.biases_std, [1, size(l.filters, 4)]);
            end
            % linear response (mean) %% TODO: concat
            %size(l.filters)
            res(i+1).x = vl_nnconv(cat(3, res(i).x, style), l.filters, l.biases, 'pad', l.pad, 'stride', l.stride);
           
            % linear response (std)
            res(i+1).s = vl_nnconv(res(i).x, l.filters_std, l.biases_std, 'pad', l.pad, 'stride', l.stride);
            res(i+1).s = exp(0.5*res(i+1).s);
                 case 'gaussian_sample',
            % gaussian sampling
            
            res(i+1).x = bsxfun(@plus, res(i).x, bsxfun(@times, res(i).s, noise));
            res(i+1).x = reshape(res(i+1).x, size(res(i).s, 1), size(res(i).s, 2), size(res(i).s, 3), size(res(i).s, 4)*l.nsample);
            
        case 'fc_mix'
            % concatenation + fc
            res(i).x = cat(3, res(i).x, repmat(style, [1 1 1 l.nsample]));
            res(i+1).x = vl_nnconv(res(i).x, l.filters, l.biases, 'pad', l.pad, 'stride', l.stride);
            
        case 'tensor_mix'
            % tensorprod + fc
            res(i+1).x = matconvnet_nntensorprod(res(i).x, repmat(style, [1 1 1 l.nsample]), ...
              l.filters, l.biases);

        case 'pool',
            % pooling
            res(i+1).x = vl_nnpool(res(i).x, l.pool, 'pad', l.pad, 'stride', l.stride, 'method', l.method);
            
        case 'unpool',
            % unpooling
            res(i+1).x = vl_nnunpool(res(i).x, 'stride', l.stride); 
        case 'reshape'
            res(i+1).x = vl_nnreshape(res(i).x, l.winsize);
            if ~isempty(res(i).s) && numel(res(i).s) > 0
              res(i+1).s = vl_nnreshape(res(i).s, l.winsize);
            end

        case 'sigmoid'
            % sigmoid
            res(i+1).x = one./(one+exp(-res(i).x));
        
        case 'tanh'
            % tanh
            res(i+1).x = matconvnet_nntanh(res(i).x);

        case 'relu'
            % rectified linear
            res(i+1).x = vl_nnrelu(res(i).x);
            
        case 'softplus'
            % softplus (smoothed relu)
            res(i+1).x = max(zero, res(i).x) + log(1+exp(-abs(res(i).x)));
            
        case 'softmax'
            % softmax (multinomial?)
            % [check] which dimension will be softmax-ed?
            res(i+1).x = vl_nnsoftmax(res(i).x);
            
        case 'normalize',
            % contrast normalization
            res(i+1).x = vl_nnnormalize(res(i).x, l.param);
            
        case 'bernoulli_ll',
            % sigmoid
            res(i).x = one./(one+exp(-res(i).x));
            res(i).x = max(min(res(i).x, 1-1e-7), 1e-7); % for numerical stability
            
            % loss for binary input
            nsample = numel(res(i).x)/numel(l.data);
            res(i).x = reshape(res(i).x, [size(l.data), nsample]);
            res(i+1).x = sum(sum(sum(sum(sum(bsxfun(@times, l.data, log(res(i).x)) + bsxfun(@times, 1-l.data, log(1-res(i).x)), 1), 2), 3), 4), 5)/nsample;
            
        case 'gaussian_ll',
            % loss for real input
            d = numel(res(i).x)/size(l.data, 4)/l.nsample;
          
            res(i).x = reshape(res(i).x, [size(l.data), l.nsample]);
            res(i).s = reshape(res(i).s, [size(l.data), l.nsample]);
            
            res(i+1).x = 0.5*d*log(2*pi);
            res(i+1).x = res(i+1).x + sum(sum(sum(log(res(i).s), 1), 2), 3);
            res(i+1).x = -sum(sum(res(i+1).x + 0.5*sum(sum(sum(bsxfun(@rdivide, bsxfun(@minus, l.data, res(i).x).^2, res(i).s.^2), 1), 2), 3), 4), 5)/l.nsample;
            
        case 'softmax_loss'
            % loss for multinomial input
            res(i+1).x = vl_nnsoftmaxloss(res(i).x, l.class);
            
        case 'gaussian_kl',
            % KL divergence
            res(i+1).x = -single(0.5)*sum(sum(sum(sum(res(i).x.^2 + res(i).s.^2 - 1 - 2*log(res(i).s), 1), 2), 3), 4);
            
        case 'dropout'
            % dropout
            if opts.disableDropout
                res(i+1).x = res(i).x;
            elseif opts.freezeDropout
                [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate, 'mask', res(i+1).aux);
            else
                [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate);
            end
            
        otherwise
            error('Unknown layer type %s', l.type);
    end
    
    if gpuMode && opts.sync
        % This should make things slower, but on MATLAB 2014a it is necessary
        % for any decent performance.
        wait(gpuDevice);
    end
    
    res(i).time = toc(res(i).time);
end


return;
