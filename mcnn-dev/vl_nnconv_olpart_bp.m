function [dzdX, dzdF, dzdB] = vl_nnconv_olpart_bp(X, F, B, dzdY, varargin)

opts = struct;
opts.pad = [0 0 0 0];
opts.stride = 1;
opts.group = [3 3];

opts = vl_argparse(opts, varargin);

numPartX = opts.group(1);
numPartY = opts.group(2);
numGroups = numPartX *numPartY;
Height = size(X, 1);
Width = size(X, 2);
Ich = size(F, 3);
Och = size(F, 4);

Height_out = size(dzdY, 1);
Width_out = size(dzdY, 2);

%B = reshape(B, [1 1 1 Och]);
dzdX = zeros(size(X, 1), size(X, 2), Ich, size(X, 4), 'single');
dzdX = cpu2gpu_copy(dzdX, isa(X, 'gpuArray'));

dzdF = 0 * F;
dzdB = 0 * B;

for i = 1:numGroups,
  part_x = floor((i-1) / numPartX) + 1;
  part_y = mod(i-1, numPartY) + 1;

  %fin_range = [((i-1) * Ich / numGroups + 1):(i*Ich/numGroups)];
  fout_range = [((i-1) * Och / numGroups + 1):(i*Och/numGroups)];

  x_range = [((part_x-1) * Height / (numPartX + 1))+1:((part_x+1)*Height / (numPartX + 1))];
  y_range = [((part_y-1) * Width / (numPartY + 1))+1:((part_y+1)*Width / (numPartY + 1))];

  xout_range = [((part_x-1) * Height_out / (numPartX + 1))+1:((part_x+1)*Height_out / (numPartX+1))];
  yout_range = [((part_y-1) * Width_out / (numPartY + 1))+1:((part_y+1)*Width_out / (numPartY+1))];

  [temp_dzdX, dzdF(:,:,:,fout_range), dzdB(fout_range)] = ...
    vl_nnconv(X(x_range, y_range, :, :), F(:,:,:, fout_range), ...
      B(fout_range), dzdY(xout_range, yout_range,:, :), 'pad', opts.pad, ...
      'stride', opts.stride);
  dzdX(x_range, y_range, :, :) = dzdX(x_range, y_range, :, :) + temp_dzdX;
end

%dzdB = reshape(dzdB, [1 1 1 Och]);

end
