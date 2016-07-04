function info = vae_info_init(opts, info, state)
% state = 0: initialize at the beginning
% state = 1: initialize every step

if state == 0,

  info.train.objective = [];
%  info.train.lb = [];
  info.train.kl = [];
  info.train.ll = [];
  info.train.speed = [];
  info.val.objective = [];
%  info.val.lb = [];
  info.val.kl = [];
  info.val.ll = [];
  info.val.speed = [];
 
elseif state == 1,
  
  info.train.objective(end+1) = 0;
%  info.train.lb(end+1) = 0;
  info.train.kl(end+1) = 0;
  info.train.ll(end+1) = 0;
  info.train.speed(end+1) = 0;
  info.val.objective(end+1) = 0;
%  info.val.lb(end+1) = 0;
  info.val.kl(end+1) = 0;
  info.val.ll(end+1) = 0;
  info.val.speed(end+1) = 0;
end



end
