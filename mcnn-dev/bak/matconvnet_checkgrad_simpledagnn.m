function [] = matconvnet_checkgrad_simpledagnn(useGpu)

addpath(genpath('/mnt/neocortex/scratch/xcyan/matconvnet/'));
vl_setupnn;

opts = struct;
opts.useGpu = useGpu;
opts.errorType = 'binary';
opts.weightDecay = 0.5;
opts.conserveMemory = false;
opts.sync = true;

rng(0);
nw = 3;
nh = 3;
nc = 1;
M = 10;
imdb.images.data = single(randn([nw, nh, nc, M]));
imdb.images.labels = (single(randn([nw, nh, nc, M])) > 0.0)+1;

f = 10.0;
net.layers = {};
% layer 1
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(3,3,10,2,'single'),...
                           'biases', randn(1,2,'single'), ...
                           'stride', 1, ...
                           'pad', 1, ...
                           'in', [2], ...
                           'out', [4]);
%% layer 2
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(3,3,1,10,'single'),...
                           'biases', randn(1,10,'single'),...
                           'stride', 1, ...
                           'pad', 1, ...
                           'in', [], ...
                           'out', [1]);
% layer 3
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(3,3,1,2,'single'), ...
                           'biases', randn(1,2,'single'),...
                           'stride', 1, ...
                           'pad', 1, ...
                           'in', [], ...
                           'out', [4]);
% layer 4
net.layers{end+1} = struct('type', 'sum', ...
                           'in', [1 3], ...
                           'out', [5]);
% layer 5
net.layers{end+1} = struct('type', 'softmax', ...
                           'in', [4], ...
                           'out', [6]);
% layer 6
net.layers{end+1} = struct('type', 'loss', ...
                           'in', [5], ...
                           'out', []);


imdb.images.data = bsxfun(@minus, imdb.images.data, mean(imdb.images.data,4));
if opts.useGpu
  %imdb.images.data = gpuArray(imdb.images.data);
  net = vl_simplenn_move(net, 'gpu');
end

info.train.objective = [];
info.train.error = [];
info.train.speed = [];

%%
res = [];
info.train.objective(end+1) = 0;
info.train.error(end+1) = 0;
info.train.speed(end+1) = 0;

im = imdb.images.data;
labels = imdb.images.labels;

if opts.useGpu,
  im = gpuArray(single(im));
  %labels = gpuArray(single(labels));
end

%labels = imdb.images.labels;

theta = matconvnet_roll_pars(net);

l2reg = 0.5;

[cost, grad] = matconvnet_cost_dagnn(theta, opts, net, im, labels, l2reg);

numGrad = computeNumericalGradient( @(x) matconvnet_cost_dagnn(x, opts, net, im, labels, l2reg), theta);
diff = norm(numGrad - grad)/norm(numGrad + grad);
fprintf('diff=%g\n',diff);
%[grad';numGrad']

end
