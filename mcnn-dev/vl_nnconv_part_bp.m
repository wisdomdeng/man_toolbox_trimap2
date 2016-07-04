function [dzdX, dzdF, dzdB] = vl_nnconv_part_bp(X, F, B, dzdY, varargin)

opts = struct;
opts.pad = [0 0 0 0];
opts.stride = 1;
opts.group = [3 3];

opts = vl_argparse(opts, varargin);

if numel(opts.pad) == 1,
  opts.pad = repmat(opts.pad, [1 4]);
elseif numel(opts.pad) == 2, 
  opts.pad = reshape(repmat(opts.pad, [2 1]), [1 4]);
end

padH = opts.pad(1) + opts.pad(3);
padW = opts.pad(2) + opts.pad(4);

numGroups = opts.group(1) * opts.group(2);
numPart = opts.group(1);
fsize = size(F, 1);
Ich = size(F, 3);
Och = size(F, 4);
Insize = size(X, 1);
Outsize = (Insize - fsize + padH + 1) / opts.stride(1);

%B = reshape(B, [1 1 1 Och]);
dzdX = zeros(size(X, 1), size(X, 2), Ich, size(X, 4), 'single');
dzdX = cpu2gpu_copy(dzdX, isa(X, 'gpuArray'));

dzdF = 0 * F;
dzdB = 0 * B;

%%
Instep = floor(Insize / opts.group(1));

InRange = zeros(opts.group(1), 2); % s, e
OutRange = zeros(opts.group(1), 2); % s, e
Padding = zeros(opts.group(1), 4); % l, r, t, b

[InRange, Padding, OutRange] = local_conv_helper(numPart, Insize, Instep, fsize, opts.pad);

%% run vl_nnconv
for i = 1:numGroups,
  part_h = floor((i-1) / numPart) + 1;
  part_w = mod(i-1, numPart) + 1;

  %fin_range = [((i-1) * Ich / numGroups + 1):(i*Ich/numGroups)];
  fout_range = [((i-1) * Och / numGroups + 1):(i*Och/numGroups)];

  x_range = [InRange(part_h,1):InRange(part_h,2)];
  y_range = [InRange(part_w,1):InRange(part_w,2)];

  xout_range = [OutRange(part_h,1):OutRange(part_h,2)];
  yout_range = [OutRange(part_w,1):OutRange(part_w,2)];

  cur_pad = [Padding(part_h,1), Padding(part_h,2), Padding(part_w, 3), Padding(part_w, 4)];

  [dzdX(x_range, y_range,:,:), dzdF(:,:,:,fout_range), dzdB(fout_range)] = ...
    vl_nnconv(X(x_range, y_range, :, :), F(:,:,:, fout_range), ...
      B(fout_range), dzdY(xout_range, yout_range,:, :), 'pad', cur_pad, ...
      'stride', opts.stride);

end

end
