function Y = vl_nnconv_olpart_ff(X, F, B, varargin)

opts = struct;
opts.pad = [0 0 0 0];
opts.stride = 1;
opts.group = [3 3];

opts = vl_argparse(opts, varargin);

numPartX = opts.group(1);
numPartY = opts.group(2);
numGroups = numPartX * numPartY;
Height = size(X, 1);
Width = size(X, 2);
Ich = size(F, 3);
Och = size(F, 4);

% assum dim(stride) = 1 
Y = zeros(size(X, 1) / opts.stride(1), size(X, 2) / opts.stride(1), Och / numGroups,  size(X, 4), 'single');
Y = cpu2gpu_copy(Y, isa(X,'gpuArray'));

Height_out = size(Y, 1);
Width_out = size(Y, 2);

%B = reshape(B, [1 1 1 Och]);

for i = 1:numGroups
  part_x = floor((i-1) / numPartX) + 1;
  part_y = mod(i-1, numPartY) + 1;
  
  %fin_range = [((i-1) * Ich / numGroups + 1):(i*Ich/numGroups)];
  fout_range = [((i-1) * Och / numGroups + 1):(i*Och/numGroups)];
  
  x_range = [((part_x-1) * Height / (numPartX + 1))+1:((part_x + 1)*Height / (numPartX + 1))];
  y_range = [((part_y-1) * Width / (numPartY + 1))+1:((part_y + 1)*Width / (numPartY + 1))];

  xout_range = [((part_x-1) * Height_out / (numPartX + 1))+1:((part_x + 1)*Height_out / (numPartX + 1))];
  yout_range = [((part_y-1) * Width_out / (numPartY + 1))+1:((part_y + 1)*Width_out / (numPartY + 1))];

  %size(X)
  %[x_range(1), x_range(end)]
  tmpY = vl_nnconv(X(x_range, y_range, :, :), F(:,:,:,fout_range), ...
    B(fout_range), 'pad', opts.pad, 'stride', opts.stride);
  
  Y(xout_range, yout_range, :, :) = Y(xout_range, yout_range, :, :) + tmpY;
end

end
