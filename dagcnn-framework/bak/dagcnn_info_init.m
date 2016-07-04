function info = dag_info_init(opts, info, state)
% state = 0: initialize at the beginning
% state = 1: initialize every step

if state == 0,

  info.train.objective = [];
  info.train.error = [];
  info.train.speed = [];
  info.val.objective = [];
  info.val.error = [];
  info.val.speed = [];
  if strcmp(opts.errorType,'binary'),
    info.train.ap = [];
    info.train.iou = [];
    info.val.ap = [];
    info.val.iou = [];
  elseif strcmp(opts.errorType,'multiclass'),
    info.train.class_error = [];
    info.val.class_error = [];
  elseif strcmp(opts.errorType, 'recon'),
    info.train.sat = [];
    info.train.sat2 = [];
    info.val.sat = [];
    info.val.sat2 = [];
  end

elseif state == 1,
  
  info.train.objective(end+1) = 0;
  info.train.error(end+1) = 0;
  info.train.speed(end+1) = 0;
  info.val.objective(end+1) = 0;
  info.val.error(end+1) = 0;
  info.val.speed(end+1) = 0;
  if strcmp(opts.errorType,'binary'),
    info.train.ap(end+1) = 0;
    info.train.iou(end+1) = 0;
    info.val.ap(end+1) = 0;
    info.val.iou(end+1) = 0;
  elseif strcmp(opts.errorType, 'multiclass'),
    info.train.class_error(end+1) = 0;
    info.val.class_error(end+1) = 0;
  elseif strcmp(opts.errorType,'recon'),
    info.train.sat(end+1) = 0;
    info.train.sat2(end+1) = 0;
    info.val.sat(end+1) = 0;
    info.val.sat2(end+1) = 0;
  end

end



end
