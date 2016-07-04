function dagcnn_showres(opts, net, spec, imdb, getBatch)

figure;
nl = numel(net.layers);
nin = numel(imdb.din.layers);
nout = numel(imdb.dout.layers);

for t = 1:opts.batchSize:size(imdb.din.layers{1}.x,4),
  batch = t:min(t+opts.batchSize-1, size(imdb.din.layers{1}.x,4));
  [din, dout] = getBatch(imdb, batch);
  for i = 1:nin,
    din.layers{i}.x = cpu2gpu_copy(din.layers{i}.x, opts.useGPU);
  end

  for i = 1:nout,
    net.layers{nl-nout+i}.class = cpu2gpu_copy(dout.layers{i}.x, opts.useGPU);
    if isfield(dout.layers{i},'mask'),
      net.layers{nl-nout+i}.mask = cpu2gpu_copy(dout.layers{i}.mask, opts.useGPU);
    end
  end

  res = matconvnet_dagnn(net, spec, din, [], [], ...
    'conserveMemory', opts.conserveMemory, ...
    'sync', opts.sync);

  pred = gather(res(end-1).x);
  gt = gather(dout.layers{1}.x);
  clear res;
  size(gt)
  for i = 1:size(pred, 4),
    pred_img = squeeze(pred(:,:,:,i));
    [max(pred_img(:)) min(pred_img(:)) mean(pred_img(:))]
    gt_img = squeeze(gt(:,:,:,i));

%    if opts.plus05
%      pred_img = pred_img + 0.5;
%      gt_img = gt_img + 0.5;
%    end
    % TODO
    if isfield(dout.layers{1},'mask')
      pred_img = bsxfun(@times, pred_img, gather(dout.layers{1}.mask(:,:,:,i)));
    end

    pred_img = imresize(pred_img, [256 256], 'bilinear');
    gt_img = imresize(gt_img, [256 256], 'bilinear');

    subplot(1,2,1), imshow(gt_img ), title('ground-truth');
    subplot(1,2,2), imshow(pred_img ), title('recon');
    drawnow;
    pause;
  end

end

end
