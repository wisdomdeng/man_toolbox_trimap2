function info = condrbm_info_init(opts, info, mode)

switch mode
case 0
  info.train.error = [];
  info.train.speed = [];
  info.train.sparsity = [];
  info.train.saturation = [];

  info.val.error = [];
  info.val.speed = [];
  info.val.sparsity = [];
  info.val.saturation = [];
case 1
  info.train.error(end+1) = 0;
  info.train.speed(end+1) = 0;
  info.train.sparsity(end+1) = 0;
  info.train.saturation(end+1) = 0;

  info.val.error(end+1) = 0;
  info.val.speed(end+1) = 0;
  info.val.sparsity(end+1) = 0;
  info.val.saturation(end+1) = 0;
end

end
