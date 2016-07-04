function info = cnn_info_init(opts, info, state)
% state = 0: initialize at the beginning
% state = 1: initialize every step

if state == 0,

  info.train.objective = [];
  info.train.error = [];
  info.train.speed = [];
  info.val.objective = [];
  info.val.error = [];
  info.val.speed = [];
  switch opts.errorType
  case 'binary'
    info.train.ap = [];
    info.train.iou = [];
    info.val.ap = [];
    info.val.iou = [];
  case 'multiclass'
    info.train.cls_err = [];
    info.val.cls_err = [];
  end

elseif state == 1,
  
  info.train.objective(end+1) = 0;
  info.train.error(end+1) = 0;
  info.train.speed(end+1) = 0;
  info.val.objective(end+1) = 0;
  info.val.error(end+1) = 0;
  info.val.speed(end+1) = 0;
  switch opts.errorType,
  case 'binary',
    info.train.ap(end+1) = 0;
    info.train.iou(end+1) = 0;
    info.val.ap(end+1) = 0;
    info.val.iou(end+1) = 0;
  case 'multiclass'
    info.train.cls_err(end+1) = 0;
    info.val.cls_err(end+1) = 0;
  end

end



end
