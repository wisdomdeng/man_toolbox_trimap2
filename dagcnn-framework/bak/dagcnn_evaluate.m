function dagcnn_evaluate(opts, net, spec, imdb, getBatch)

nl = numel(net.layers);
nin = numel(imdb.din.layers);
nout = numel(imdb.dout.layers);

tot = struct;
tot.avgError = [];
tot.sat = [];
tot.sat2 = [];

for t = 1:opts.batchSize:size(imdb.din.layers{1}.x,4),
  t 
  batch = t:min(t+opts.batchSize-1, size(imdb.din.layers{1}.x,4));
  [din, dout] = getBatch(imdb, batch);

  for i = 1:nin
    din.layers{i}.x = cpu2gpu_copy(din.layers{i}.x, opts.useGPU);
  end

  for i = 1:nout,
    net.layers{nl-nout+1}.class = cpu2gpu_copy(dout.layers{i}.x, opts.useGPU);
    if isfield(dout.layers{i},'mask'),
      net.layers{nl-nout+i}.mask = cpu2gpu_copy(dout.layers{i}.mask, opts.useGPU);
    end
  end

  res = matconvnet_dagnn(net, spec, din, [], [], ...
    'conserveMemory', opts.conserveMemory, ...
    'sync', opts.sync);

  pred = gather(res(end-1).x);
  gt = gather(dout.layers{1}.x); %% TODO
  mask = gather(net.layers{end}.mask);
  clear res;

  cur = struct;
  for i = 1:size(pred, 4),
    cur.pred = squeeze(pred(:,:,:,i));
    cur.gt = squeeze(gt(:,:,:,i));
    cur.mask = squeeze(mask(:,:,:,i));
    
    cur.error = bsxfun(@times, bsxfun(@minus, cur.gt, cur.pred), cur.mask).^2;
    cur.point_error = sum(cur.error, 3);
    cur.sat = sum(cur.point_error(:)<0.1 & cur.mask(:) == 1)/sum(cur.mask(:) == 1);
    cur.sat2 = sum(cur.point_error(:)<0.01 & cur.mask(:) == 1)/sum(cur.mask(:) == 1);
    cur.avgError = sum(cur.error(:));

    tot.avgError = cat(1, tot.avgError, cur.avgError);
    tot.sat = cat(1, tot.sat, cur.sat);
    tot.sat2 = cat(1, tot.sat2, cur.sat2);
  end

  fprintf('test avgError = %.4f \t sat = %.4f \t sat2 = %.4f\n', ...
    mean(tot.avgError), mean(tot.sat), mean(tot.sat2));
end

end
