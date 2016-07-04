function cnn_evaluate(opts, net, imdb, getBatch)

% preprocess
% TODO
%imdb.images.data(:,:,1:3,:) = bsxfun(@minus, imdb.images.data(:,:,1:3,:), mean(imdb.images.data(:,:,1:3,:), 4));

num_test = size(imdb.images.data, 4);

tot = struct;
tot.avgPrec = [];
tot.iouScore = [];
for t = 1:opts.batchSize:num_test,
  t 
  batch = t:min(t+opts.batchSize-1, num_test);
  [im, label] = getBatch(imdb, batch);

  %
  net.layers{end}.class = gpuArray(single(label));
  res = matconvnet_simplenn(net, gpuArray(single(im)), [], []);

  ypred = gather(res(end-1).x);
  clear res;

  cur = struct;
  for i = 1:size(im, 4),
    cur.ypred = squeeze(ypred(:,:,1,i));
    cur.ygt = bsxfun(@minus, 2, label(:,:,i));

    error = ((cur.ypred(:) > 0.5) ~= cur.ygt(:));
    cur.avgPrec = 1. - mean(error(:));
    cur.iouScore = compute_iou(cur.ypred(:)>0.5, cur.ygt(:));
    tot.avgPrec = cat(1, tot.avgPrec, cur.avgPrec);
    tot.iouScore = cat(1, tot.iouScore, cur.iouScore);
    
  end

  fprintf('test avgPrec = %.4f \t iouScore = %.4f\n', ...
    mean(tot.avgPrec), mean(tot.iouScore));
end

end
