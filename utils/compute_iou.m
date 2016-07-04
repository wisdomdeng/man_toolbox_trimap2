function [iouscore] = compute_iou(pred, gt)

if sum(pred) == 0 && sum(gt) == 0,
  iouscore = 0.0;
  return;
end

iouscore = sum(pred.*gt)./sum(pred|gt);

end
