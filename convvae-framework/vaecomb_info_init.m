function info = vaecomb_info_init(opts, info, state)
% state = 0: initialize at the beginning
% state = 1: initialize every step

if state == 0,

  info = vae_info_init(opts, info, 0);
  info.train.gc = [];
  info.val.gc = [];

elseif state == 1,

  info = vae_info_init(opts, info, 1);
  info.train.gc(end+1) = 0;
  info.val.gc(end+1) = 0;
end

end
