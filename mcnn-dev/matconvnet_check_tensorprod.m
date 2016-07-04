function [] = matconvnet_checkgrad_tensorprod(useGPU)

%addpath(genpath('/mnt/neocortex/scratch/xcyan/matconvnet-1.0-beta9/'));
%vl_setupnn;
%addpath(genpath('/mnt/neocortex/scratch/xcyan/xcyan_toolbox/'));
dagcnn_config;

opts = struct;
opts.useGPU = useGPU;
opts.errorType = 'recon';
opts.weightDecay = 0.5;
opts.conserveMemory = false;
opts.sync = true;

rng(0);
nw = 1;
nh = 1;
nc = 4;
nc2 = 3;
nc3 = 2;
nf = 5;
M = 3;

im1 = randn([nw, nh, nc, M]);
im1 = cpu2gpu_copy(im1, useGPU);
im2 = randn([nw, nh, nc2, M]);
im2 = cpu2gpu_copy(im2, useGPU);
ou1 = randn([nw, nh, nc3, M]);
ou1 = cpu2gpu_copy(ou1, useGPU);
one = cpu2gpu_copy(1.0, useGPU);

%% data in and out
din = struct;
din.layers = {};
din.layers{end+1} = struct('name','data1','x',im1);
din.layers{end+1} = struct('name','data2','x',im2);

dout = struct;
dout.layers = {};
dout.layers{end+1} = struct('name','output','x',ou1,'dzdy',one);

% network structure
f = 10.0;
net.layers = {};
net.layers{end+1} = struct('type', 'sigmoid', 'name','sm1');
net.layers{end+1} = struct('type', 'relu', 'name','sm2');

net.layers{end+1} = struct('type', 'tensorprod', ...
                           'filters', f*randn(1,1,nc+nc2+nc3,nf,'single'),...
                           'biases',randn(1, nc3,'single'), ...
                           'name','tprod');

net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(1,1,nc3,nc3,'single'),...
                           'biases', randn(1, nc3,'single'), ...
                           'stride', 1, 'pad', 0, ...
                           'name', 'conv');
net.layers{end+1} = struct('type', 'euloss', 'name', 'loss');

net.layers{1}.in = []; net.layers{1}.out = {'tprod'};
net.layers{2}.in = []; net.layers{2}.out = {'tprod'};
net.layers{3}.in = {'sm1','sm2'}; net.layers{3}.out = {'conv'};
net.layers{4}.in = {'tprod'}; net.layers{4}.out = {'loss'};
net.layers{5}.in = {'conv'}; net.layers{5}.out = [];

%% spec
spec = matconvnet_createSpec(net);

if opts.useGPU,
  net = matconvnet_dagnn_move(net, 'gpu');
end

theta = matconvnet_roll_pars(net);

l2reg = 0.5;

[cost, grad] = matconvnet_cost_dagnn(theta, opts, net, spec, din, dout, l2reg);

numGrad = computeNumericalGradient( @(x) matconvnet_cost_dagnn(x, ...
  opts, net, spec, din, dout, l2reg), theta);
%size(numGrad)
%size(grad)
diff = norm(numGrad - grad)/norm(numGrad + grad);
fprintf('diff=%g\n',diff);
%[grad,numGrad]

end
