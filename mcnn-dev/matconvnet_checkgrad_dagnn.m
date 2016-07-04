function [] = matconvnet_checkgrad_dagnn(useGPU, testcase_id)

cnn_config;

switch testcase_id
case 1
  do_test1(useGPU);
case 2
  do_test2(useGPU);
end

%sprintf('[%s]\n', net.layers{1}.name);
end % end of main function

%% ==========================
% testcase_1
function do_test1(useGPU)

fprintf('Start running do_test1...\n');

opts = struct;
opts.useGPU = useGPU;
opts.errorType = 'recon';
opts.weightDecay = 0.5;
opts.conserveMemory = false;
opts.sync = true;

rng(0);
nw = 5;
nh = 7;
nc = 1;
M = 10;

im1 = randn([nw, nh, nc, M]);
im1 = cpu2gpu_copy(im1, useGPU);
im2 = randn([nw, nh, nc, M]);
im2 = cpu2gpu_copy(im2, useGPU);
ou1 = randn([nw, nh, 2, M]);
ou1 - cpu2gpu_copy(ou1, useGPU);
one = 1.0;
one = cpu2gpu_copy(one, useGPU);

%% data in and out
din = struct;
din.layers = {};
din.layers{end+1} = struct('name', 'data1', ...
                           'x', im1);
din.layers{end+1} = struct('name', 'data2', ...
                           'x', im2);


dout = struct;
dout.layers = {};
dout.layers{end+1} = struct('name', 'output', ...
                            'x', ou1, ...
                            'dzdy', one);

%% network structure
f = 10.0;
net = struct;
net.layers = {};
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(3,3,1,4,'single'), ...
                           'biases', randn(1,4,'single'),...
                           'stride', 1, ...
                           'pad', 1, ...
                           'name', 'conv1');

net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(3,3,1,6,'single'), ...
                           'biases', randn(1,6,'single'), ...
                           'stride', 1, ...
                           'pad', 1, ...
                           'name', 'conv2');

net.layers{end+1} = struct('type', 'relu', ...
                           'name', 'relu1');

net.layers{end+1} = struct('type', 'concat', ...
                           'name', 'concat1');

net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(3,3,10,2,'single'), ...
                           'biases', randn(1,2,'single'), ...
                           'stride', 1, ...
                           'pad', 1, ...
                           'name', 'conv3');

net.layers{end+1} = struct('type', 'euloss', ...
                           'name', 'loss1');

net.layers{1}.in = []; net.layers{1}.out = {'relu1'};
net.layers{2}.in = []; net.layers{2}.out = {'concat1'};
net.layers{3}.in = {'conv1'}; net.layers{3}.out = {'concat1'};
net.layers{4}.in = {'relu1', 'conv2'}; net.layers{4}.out = {'conv3'};
net.layers{5}.in = {'concat1'}; net.layers{5}.out = {'loss1'};
net.layers{6}.in = {'conv3'}; net.layers{6}.out = [];

%% din1 --> conv1 --> relu1 --+
%%                            +-> concat1 --> conv3 --> RMSEloss --> dout1
%% din2 --> conv2 ------------+


%% spec
spec = matconvnet_createSpec(net); %% TODO

if opts.useGPU,
  net = matconvnet_dagnn_move(net, 'gpu');
end

theta = matconvnet_roll_pars(net);

l2reg = 0.5;

%% TODO
[cost, grad] = matconvnet_cost_dagnn(theta, opts, net, spec, din, dout, l2reg);

numGrad = computeNumericalGradient( @(x) matconvnet_cost_dagnn(x, ...
  opts, net, spec, din, dout, l2reg), theta);

diff = norm(numGrad - grad)/norm(numGrad + grad);
fprintf('diff=%g\n', diff);
[numGrad, grad]

end
