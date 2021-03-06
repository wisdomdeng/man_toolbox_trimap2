function res = vae_bp(net, noise, dzdy, res, varargin)

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
% -------------------------------------------------------------------------

for i = n:-1:1,
    l = net.layers{i};
    res(i).backwardTime = tic;
    
    switch l.type
        case 'early_split',
            % linear (convolutional) response
            [res(i).dzdx, res(i).filters, res(i).biases] = ...
                vl_nnconv(res(i).x, l.filters, l.biases, ...
                res(i+1).dzdx, 'pad', l.pad, 'stride', l.stride);
            
            [res(i).dzds, res(i).filters_std, res(i).biases_std] = ...
                vl_nnconv(res(i).x, l.filters_std, l.biases_std, ...
                res(i+1).dzds, 'pad', l.pad, 'stride', l.stride);
            
            res(i).dzdx = res(i).dzdx + res(i).dzds;

        case 'conv_streams'
            % linear (convolutional) response
            [res(i).dzdx, res(i).filters, res(i).biases] = ...
                vl_nnconv(res(i).x, l.filters, l.biases, ...
                res(i+1).dzdx, 'pad', l.pad, 'stride', l.stride);
            
            [res(i).dzds, res(i).filters_std, res(i).biases_std] = ...
                vl_nnconv(res(i).s, l.filters_std, l.biases_std, ...
                res(i+1).dzds, 'pad', l.pad, 'stride', l.stride);
        
        case 'conv'

            [res(i).dzdx, res(i).filters, res(i).biases] = ...
                vl_nnconv(res(i).x, l.filters, l.biases, ...
                res(i+1).dzdx, 'pad', l.pad, 'stride', l.stride);

        case 'conv_part'
            
            [res(i).dzdx, res(i).filters, res(i).biases] = ...
                vl_nnconv_part_bp(res(i).x, l.filters, l.biases, ...
                res(i+1).dzdx, 'pad', l.pad, 'stride', l.stride, 'group', l.group);

        case 'conv_olpart'

           [res(i).dzdx, res(i).filters, res(i).biases] = ...
                vl_nnconv_olpart_bp(res(i).x, l.filters, l.biases, ...
                res(i+1).dzdx, 'pad', l.pad, 'stride', l.stride, 'group', l.group);


        case 'conv_gaussian',
            if ~isfield(l, 'share_std'),
              l.share_std = 0;
            end

            if l.share_std == 1, % spatial
              l.filters_std = repmat(l.filters_std, [size(l.filters, 1), size(l.filters, 2), 1, 1]);
            elseif l.share_std == 2, % spatial + channel
              l.filters_std = repmat(l.filters_std, [size(l.filters, 1), size(l.filters, 2), 1, size(l.filters, 4)]);
              l.biases_std = repmat(l.biases_std, [1, size(l.filters, 4)]);
            elseif l.share_std == 3, % channel
              l.filters_std = repmat(l.filters_std, [1, 1, 1, size(l.filters, 4)]);
              l.biases_std = repmat(l.biases_std, [1, size(l.filters, 4)]);
            end
            
            % linear (conv) response for mean
            [res(i).dzdx, res(i).filters, res(i).biases] = ...
                vl_nnconv(res(i).x, l.filters, l.biases, ...
                res(i+1).dzdx, 'pad', l.pad, 'stride', l.stride);
            
            % linear (conv) response for std
            if ~isempty(res(i).s) && numel(res(i).s) > 0,
              [res(i).dzds, res(i).filters_std, res(i).biases_std] = ...
                vl_nnconv(res(i).s, l.filters_std, l.biases_std, ...
                res(i+1).dzds, 'pad', l.pad, 'stride', l.stride);
            else
              [res(i).dzds, res(i).filters_std, res(i).biases_std] = ...
                vl_nnconv(res(i).x, l.filters_std, l.biases_std, ...
                res(i+1).dzds, 'pad', l.pad, 'stride', l.stride);
            end

            if l.share_std == 1, 
              res(i).filters_std = sum(sum(res(i).filters_std, 1), 2);
            elseif l.share_std == 2,
              res(i).filters_std = sum(sum(sum(res(i).filters_std, 1), 2), 4);
              res(i).biases_std = sum(res(i).biases_std, 2);
            elseif l.share_std == 3,
              res(i).filters_std = sum(res(i).filters_std, 4);
              res(i).biases_std = sum(res(i).biases_std, 2);
            end

            if ~isempty(res(i).s) && numel(res(i).s) > 0,
              res(i).dzds = res(i).dzds;
            else
              res(i).dzdx = res(i).dzdx + res(i).dzds;
            end

        case 'gaussian_sample',
            % gaussian sampling
            res(i+1).dzdx = reshape(res(i+1).dzdx, [size(res(i).s), l.nsample]);
            res(i).dzds = 0.5*sum(res(i+1).dzdx.*noise, 5).*res(i).s;
            res(i).dzdx = sum(res(i+1).dzdx, 5);
            
        case 'pool',
            % pooling
            res(i).dzdx = vl_nnpool(res(i).x, l.pool, res(i+1).dzdx, ...
                'pad', l.pad, 'stride', l.stride, 'method', l.method);
            if ~isempty(res(i).s) && numel(res(i).s) > 0,
              res(i).dzds = vl_nnpool(res(i).s, l.pool, res(i+1).dzds, ...
                  'pad', l.pad, 'stride', l.stride, 'method', l.method);
            end

        case 'unpool',
            % unpooling
            res(i).dzdx = vl_nnunpool(res(i).x, res(i+1).dzdx, 'stride', l.stride);
            
            if ~isempty(res(i).s) && numel(res(i).s) > 0,
              res(i).dzds = vl_nnunpool(res(i).s, res(i+1).dzds, 'stride', l.stride);
            end

        case 'reshape'
            res(i).dzdx = vl_nnreshape(res(i).x, l.winsize, res(i+1).dzdx);
            if ~isempty(res(i).s) && numel(res(i).s) > 0,
              res(i).dzds = vl_nnreshape(res(i).s, l.winsize, res(i+1).dzds);
            end

        %case 'sigmoid',
            % sigmoid
        %    res(i+1).x = reshape(res(i+1).x, size(res(i+1).dzdx));
        %    res(i).dzdx = res(i+1).x.*(one-res(i+1).x).*res(i+1).dzdx;
        
        case 'tanh'
            res(i).dzdx = matconvnet_nntanh(res(i).x, res(i+1).dzdx);
            if ~isempty(res(i).s) && numel(res(i).s) > 0,
              res(i).dzds = matconvnet_nntanh(res(i).s, res(i+1).dzds);
            end

        case 'relu',
            % rectified linear
            
            res(i).dzdx = vl_nnrelu(res(i).x, res(i+1).dzdx);
            if ~isempty(res(i).s) && numel(res(i).s) > 0, 
              res(i).dzds = vl_nnrelu(res(i).s, res(i+1).dzds);
            end
            
        %case 'softplus',
            % softplus (smoothed relu)
        %    res(i).dzdx = ((exp(res(i+1).x)-one)./exp(res(i+1).x)).*res(i+1).dzdx;
            
        %case 'softmax',
            % softmax (multinomial)
        %    res(i).dzdx = vl_nnsoftmax(res(i).x, res(i+1).dzdx);
            
        case 'normalize',
            % contrast normalization
            res(i).dzdx = vl_nnnormalize(res(i).x, l.param, res(i+1).dzdx);
            if ~isempty(res(i).s) && numel(res(i).s) > 0, 
              res(i).dzds = vl_nnnormalize(res(i).s, l.param, res(i+1).dzds);
            end

        case 'bernoulli_ll',
            % loss for binary input + sigmoid
            res(i).dzdx = -res(i+1).dzdx*bsxfun(@minus, res(i).x, l.data);
            res(i).dzdx = reshape(res(i).dzdx, size(l.data, 1), size(l.data, 2), size(l.data, 3), size(l.data, 4)*l.nsample)/l.nsample;

        case 'bernoulli_ll_fgbg',
            % loss for binary input + sigmoid
           
            res(i).dzdx = res(i).dzdx - res(i+1).dzdx*bsxfun(@minus, res(i).x, l.data);
            res(i).dzdx = reshape(res(i).dzdx, size(l.data, 1), size(l.data, 2), size(l.data, 3), size(l.data, 4)*l.nsample)/l.nsample;
           
        case 'tanh_gaussian_ll' %% assume res(i).s = 1
            trunc_x = max(min(res(i).x, one), -one);
            trunc_x = matconvnet_nntanh(trunc_x);

            res(i).dzdx = -res(i+1).dzdx*((bsxfun(@minus, l.data, trunc_x)>0)-0.5)*2;
            res(i).dzdx = matconvnet_nntanh(trunc_x, res(i).dzdx);
            res(i).dzdx = reshape(res(i).dzdx, size(l.data, 1), size(l.data, 2), size(l.data, 3), size(l.data, 4)*l.nsample)/l.nsample;
            

            %res(i).dzds = -0.5*(one - bsxfun(@rdivide, bsxfun(@minus, trunc_x, l.data), res(i).s).^2);
            %res(i).dzds = reshape(res(i).dzds, size(l.data, 1), size(l.data, 2), size(l.data, 3), size(l.data, 4)*l.nsample)/l.nsample;
            
        case 'gaussian_ll',
            % loss for real input
            res(i).dzdx = -res(i+1).dzdx*bsxfun(@rdivide, bsxfun(@minus, res(i).x, l.data), res(i).s.^2);
            res(i).dzdx = reshape(res(i).dzdx, size(l.data, 1), size(l.data, 2), size(l.data, 3), size(l.data, 4)*l.nsample)/l.nsample;
            
            res(i).dzds = -0.5*(one - bsxfun(@rdivide, bsxfun(@minus, res(i).x, l.data), res(i).s).^2);
            res(i).dzds = reshape(res(i).dzds, size(l.data, 1), size(l.data, 2), size(l.data, 3), size(l.data, 4)*l.nsample)/l.nsample;
            
        %case 'softmax_loss'
            % loss for multinomial input
        %    res(i).dzdx = vl_nnsoftmaxloss(res(i).x, l.class, res(i+1).dzdx);
            
        case 'gaussian_kl',
            % KL divergence
            res(i).dzdx = res(i+1).dzdx - res(i).x;
            res(i).dzds = res(i+1).dzds - single(0.5)*(res(i).s.^2 - one);
            
        case 'dropout'
            if opts.disableDropout
                res(i).dzdx = res(i+1).dzdx;
                if ~isempty(res(i).s) && numel(res(i).s) > 0,
                  res(i).dzds = res(i+1).dzds;
                end
            else
              res(i).dzdx = vl_nndropout(res(i).x, res(i+1).dzdx, 'mask', res(i+1).aux);
              if ~isempty(res(i).s) && numel(res(i).s) > 0,
                res(i).dzds = vl_nndropout(res(i).s, res(i+1).dzds, 'mask', res(i+1).aux);
              end
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
