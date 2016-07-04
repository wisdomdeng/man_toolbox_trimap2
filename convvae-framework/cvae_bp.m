function res = cvae_bp(net, style, noise, dzdy, res, varargin)

one = single(1);
zero = single(0);


% -------------------------------------------------------------------------
%                                                      default parameters
% -------------------------------------------------------------------------

opts.res = [];
opts.conserveMemory = false;
opts.sync = false;
opts.disableDropout = false;
opts.freezeDropout = false;

opts = vl_argparse(opts, varargin);


% -------------------------------------------------------------------------
%                                                        number of layers
% -------------------------------------------------------------------------

n = numel(net.layers);

% -------------------------------------------------------------------------
%                                                                     gpu
% -------------------------------------------------------------------------

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

if isempty(dzdy),
    gpuMode = isa(res(end).dzdx, 'gpuArray');
else
    gpuMode = isa(dzdy, 'gpuArray');
    res(n+1).dzdx = dzdy;
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

for i = n:-1:1,
    l = net.layers{i};
    res(i).backwardTime = tic;
    
    switch l.type
        case 'conv',
            % linear (convolutional) response
            [res(i).dzdx, res(i).filters, res(i).biases] = ...
                vl_nnconv(res(i).x, l.filters, l.biases, ...
                res(i+1).dzdx, 'pad', l.pad, 'stride', l.stride);
        
        case 'conv_part'
            [res(i).dzdx, res(i).filters, res(i).biases] = ...
                vl_nnconv_part_bp(res(i).x, l.filters, l.biases, ...
                res(i+1).dzdx, 'pad', l.pad, 'stride', l.stride, ...
                'group', l.group);

        case 'conv_gaussian',
            if ~isfield(l, 'share_std'),
              l.share_std = 0;
            end

            if l.share_std == 1, % spatial
              l.filters_std = repmat(l.filters_std, [size(l.filters, 1), size(l.filters, 2), 1, 1]);
            elseif l.share_std == 2, % spatial + channel
              l.filters_std = repmat(l.filters_std, [size(l.filters, 1), size(l.filters, 2), 1, size(l.filters, 4)]);
              l.biases_std = repmat(l.biases_std, [1, size(l.filters, 4)]);
            end
          
            % linear (conv) response for mean
            i 
            [res(i).dzdx, res(i).filters, res(i).biases] = ...
                vl_nnconv(res(i).x, l.filters, l.biases, ...
                res(i+1).dzdx, 'pad', l.pad, 'stride', l.stride);
            
            % linear (conv) response for std
            [res(i).dzds, res(i).filters_std, res(i).biases_std] = ...
                vl_nnconv(res(i).x, l.filters_std, l.biases_std, ...
                res(i+1).dzds, 'pad', l.pad, 'stride', l.stride);
            
            if l.share_std == 1,
              res(i).filters_std = sum(sum(res(i).filters_std, 1), 2);
            elseif l.share_std == 2,
              res(i).filters_std = sum(sum(sum(res(i).filters_std, 1), 2), 4);
              res(i).biases_std = sum(res(i).biases_std, 2);
            end

            res(i).dzdx = res(i).dzdx + res(i).dzds;
        
        case 'fc_mix_gaussian',
            if ~isfield(l, 'share_std'),
              l.share_std = 0;
            end

            if l.share_std == 1, % spatial
              l.filters_std = repmat(l.filters_std, [size(l.filters, 1), size(l.filters, 2), 1, 1]);
            elseif l.share_std == 2, % spatial + channel
              l.filters_std = repmat(l.filters_std, [size(l.filters, 1), size(l.filters, 2), 1, size(l.filters, 4)]);
              l.biases_std = repmat(l.biases_std, [1, size(l.filters, 4)]);
            end
          
            % linear (conv) response for mean
            [res(i).dzdx, res(i).filters, res(i).biases] = ...
                vl_nnconv(cat(3, res(i).x, style), l.filters, l.biases, ...
                res(i+1).dzdx, 'pad', l.pad, 'stride', l.stride);
            %% TODO: split
            res(i).dzdx = res(i).dzdx(:,:,1:end-size(style,3),:);
            
            % linear (conv) response for std
            [res(i).dzds, res(i).filters_std, res(i).biases_std] = ...
                vl_nnconv(res(i).x, l.filters_std, l.biases_std, ...
                res(i+1).dzds, 'pad', l.pad, 'stride', l.stride);
            
            if l.share_std == 1,
              res(i).filters_std = sum(sum(res(i).filters_std, 1), 2);
            elseif l.share_std == 2,
              res(i).filters_std = sum(sum(sum(res(i).filters_std, 1), 2), 4);
              res(i).biases_std = sum(res(i).biases_std, 2);
            end

            res(i).dzdx = res(i).dzdx + res(i).dzds;
         
        case 'gaussian_sample',
            % gaussian sampling
            
            res(i+1).dzdx = reshape(res(i+1).dzdx, [size(res(i).s), l.nsample]);
            res(i).dzds = 0.5*sum(res(i+1).dzdx.*noise, 5).*res(i).s;
            res(i).dzdx = sum(res(i+1).dzdx, 5);
        
        case 'fc_mix'
            % concatenation + fc
            
            [res(i).dzdx, res(i).filters, res(i).biases] = ...
                vl_nnconv(res(i).x, l.filters, l.biases, ...
                res(i+1).dzdx, 'pad', l.pad, 'stride', l.stride);
            
            res(i).dzdx = res(i).dzdx(:,:,1:end-size(style,3),:);
        case 'tensor_mix'
            [res(i).dzdx, ~, res(i).filters, res(i).biases] = ...
                matconvnet_nntensorprod(res(i).x, repmat(style, [1 1 1 l.nsample]), ...
                  l.filters, l.biases, res(i+1).dzdx);

        case 'pool',
            % pooling
            res(i).dzdx = vl_nnpool(res(i).x, l.pool, res(i+1).dzdx, ...
                'pad', l.pad, 'stride', l.stride, 'method', l.method);
            
        case 'unpool',
            % unpooling
            res(i).dzdx = vl_nnunpool(res(i).x, res(i+1).dzdx, 'stride', l.stride);

        case 'reshape'
            %% reshaping layer
            res(i).dzdx = vl_nnreshape(res(i).x, l.winsize, res(i+1).dzdx);
            if ~isempty(res(i+1).dzds) && numel(res(i+1).dzds) > 0, 
              res(i).dzds = vl_nnreshape(res(i).s, l.winsize, res(i+1).dzds);
            end

        case 'sigmoid',
            % sigmoid
            res(i+1).x = reshape(res(i+1).x, size(res(i+1).dzdx));
            res(i).dzdx = res(i+1).x.*(one-res(i+1).x).*res(i+1).dzdx;
        
        case 'tanh'
            % tanh
            res(i).dzdx = matconvnet_nntanh(res(i).x, res(i+1).dzdx);

        case 'relu',
            % rectified linear
            res(i).dzdx = vl_nnrelu(res(i).x, res(i+1).dzdx);
            
        case 'softplus',
            % softplus (smoothed relu)
            res(i).dzdx = ((exp(res(i+1).x)-one)./exp(res(i+1).x)).*res(i+1).dzdx;
            
        case 'softmax',
            % softmax (multinomial)
            res(i).dzdx = vl_nnsoftmax(res(i).x, res(i+1).dzdx);
            
        case 'normalize',
            % contrast normalization
            res(i).dzdx = vl_nnnormalize(res(i).x, l.param, res(i+1).dzdx);
            
        case 'bernoulli_ll',
            % loss for binary input + sigmoid
            res(i).dzdx = -res(i+1).dzdx*bsxfun(@minus, res(i).x, l.data);
            res(i).dzdx = reshape(res(i).dzdx, size(l.data, 1), size(l.data, 2), size(l.data, 3), size(l.data, 4)*l.nsample)/l.nsample;
            
        case 'gaussian_ll',
            % loss for real input
            res(i).dzdx = -res(i+1).dzdx*bsxfun(@rdivide, bsxfun(@minus, res(i).x, l.data), res(i).s.^2);
            res(i).dzdx = reshape(res(i).dzdx, size(l.data, 1), size(l.data, 2), size(l.data, 3), size(l.data, 4)*l.nsample)/l.nsample;
            
            res(i).dzds = -0.5*(one - bsxfun(@rdivide, bsxfun(@minus, res(i).x, l.data), res(i).s).^2);
            res(i).dzds = reshape(res(i).dzds, size(l.data, 1), size(l.data, 2), size(l.data, 3), size(l.data, 4)*l.nsample)/l.nsample;
            
        case 'softmax_loss'
            % loss for multinomial input
            res(i).dzdx = vl_nnsoftmaxloss(res(i).x, l.class, res(i+1).dzdx);
            
        case 'gaussian_kl',
            % KL divergence
            res(i).dzdx = res(i+1).dzdx - res(i).x;
            res(i).dzds = res(i+1).dzds - single(0.5)*(res(i).s.^2 - one);
            
        case 'dropout'
            if opts.disableDropout
                res(i).dzdx = res(i+1).dzdx;
            else
              res(i).dzdx = vl_nndropout(res(i).x, res(i+1).dzdx, 'mask', res(i+1).aux);
            end
            
        otherwise
            error('Unknown layer type %s', l.type);
    end
    
    if gpuMode && opts.sync,
        % This should make things slower, but on MATLAB 2014a it is necessary
        % for any decent performance.
        wait(gpuDevice);
    end
    
    res(i).backwardTime = toc(res(i).backwardTime);
end


return;
