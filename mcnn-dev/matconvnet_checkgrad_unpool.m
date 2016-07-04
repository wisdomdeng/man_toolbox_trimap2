function [] = matconvnet_checkgrad_unpool(useGpu)

addpath(genpath('/mnt/neocortex/scratch/xcyan/matconvnet/'));
vl_setupnn;

opts = struct;
opts.useGpu = useGpu;
opts.errorType = 'binary';
opts.weightDecay = 0.5;
opts.conserveMemory = false;
opts.sync = true;

rng(0);
nw = 5;
nh = 5;
nc = 1;
M = 10;
imdb.images.data = single(randn([nw, nh, nc, M]));
imdb.images.labels = (single(randn([nw, nh, nc, M])) > 0.0)+1;

f = 10.0;
net.layers = {};
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(3,3,1,2,'single'),...
                           'biases', randn(1,2,'single'),...
                           'stride', 1, ...
                           'pad', 0);
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 1, ...
                           'pad', 0);
net.layers{end+1} = struct('type', 'unpool', ...
                           'stride', [3 3]);
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(3,3,2,2,'single'), ...
                           'biases', randn(1,2,'single'), ...
                           'stride', 1, ...
                           'pad', [1 0 1 0]);
net.layers{end+1} = struct('type', 'softmax');
net.layers{end+1} = struct('type', 'loss');

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

theta = matconvnet_roll_pars(net);


l2reg = 0.5;

[cost, grad] = matconvnet_cost_mini(theta, opts, net, im, labels, l2reg);

numGrad = computeNumericalGradient( @(x) matconvnet_cost_mini(x, opts, net, im, labels, l2reg), theta);
diff = norm(numGrad - grad)/norm(numGrad + grad);
fprintf('diff=%g\n',diff);
%[grad';numGrad']

end
