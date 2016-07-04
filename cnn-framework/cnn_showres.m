function cnn_showres(opts, net)

addpath('/mnt/neocortex3/scratch/xcyan/aorta_seg/utils/');
% preprocess
% TODO
%testImgs = imdb.images.data;
%testImgs(:,:,1:3,:) = bsxfun(@minus, testImgs(:,:,1:3,:), mean(testImgs(:,:,1:3,:), 4));

%im = testImgs;
%label = imdb.images.labels;
%
%if opts.useGpu,
%  net = vl_simplenn_move(net, 'gpu');
%end

%net.layers{end}.class = gpuArray(single(label));

%res = matconvnet_simplenn(net, gpuArray(single(im)), [], []);

figure(1);
filters_lvl1 = gather(net.layers{1}.filters);
%min(net.layers{1}.filters(:))
%max(net.layers{1}.filters(:))
[ws, ~, nch, nch2] = size(filters_lvl1);
figure(1);
display_network_nonsquare(reshape(filters_lvl1(:,:,1,:), ws*ws, nch2));

%figure(2);
%for i = 1:size(im, 4),
%  subplot(1,2,1), imshow(imdb.images.data(:,:,1:3,i)), drawnow;
%  subplot(1,2,2), imshow(gather(res(end-1).x(:,:,1,i))), drawnow;
%  pause;
%end

end
