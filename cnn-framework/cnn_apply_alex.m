function ans = cnn_apply_alex()

cnn_config;
load('../data/trainImgs.mat');
trainImgs = Im_transform(trainImgs, 227);
trainImgs = single(trainImgs/255.0);
%trainImgs = bsxfun(@minus, trainImgs, mean(trainImgs, 4));

sampleX = trainImgs(:,:,:,1000:1128);
sampleX = gpuArray(sampleX);
net = load('../model/imagenet-caffe-alex.mat');
net = vl_simplenn_move(net, 'gpu');
res = matconvnet_simplenn(net, sampleX, [], []);

ranklist = squeeze(gather(res(end).x));
[val, ind] = max(ranklist,[],1);
size(ind)
ans = cell(length(ind), 1);
for i = 1:length(ind)
  ans{i}.d = net.classes.description{ind(i)};
  ans{i}.s = val(i);
end

end
