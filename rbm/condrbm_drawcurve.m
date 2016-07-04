function [info, opts] = condrbm_drawcurve(info, opts, epoch)

% visualization
info.train.error(end) = info.train.error(end) / info.numtrain;
info.train.speed(end) = info.numtrain / info.train.speed(end);
info.train.sparsity(end) = info.train.sparsity(end) / info.numtrain;
info.train.saturation(end) = info.train.saturation(end) / info.numtrain;

info.val.error(end) = info.val.error(end) / info.numval;
info.val.speed(end) = info.numval / info.val.speed(end);
info.val.sparsity(end) = info.val.sparsity(end) / info.numval;
info.val.saturation(end) = info.val.saturation(end) / info.numval;

if ~opts.verbose,
  fprintf(' train_speed = (%.1f images/s)\t val_speed = (%.1f images/s)\n', ...
    info.train.speed(end), info.val.speed(end));
  fprintf(' train_error = (%.3f)\t val_error = (%.3f)\n', ...
    info.train.error(end), info.val.error(end));
  fprintf(' train_sparsity = (%.3f)\t val_sparsity = (%.3f)\n', ...
    info.train.sparsity(end), info.val.sparsity(end));
  fprintf(' train_saturation = (%.3f)\t val_saturation = (%.3f)\n', ...
    info.train.saturation(end), info.val.saturation(end));
  fprintf('\n');
else

  figure(1); clf;
  subplot(1,2,1);
  semilogy(1:epoch, info.train.error, 'k'); hold on;
  semilogy(1:epoch, info.val.error, 'b');
  xlabel('training epoch'); ylabel('error');
  grid on;
  h=legend('train','val');
  set(h,'color','none');
  title('error-epoch curve');

  subplot(1,2,2);
  plot(1:epoch, info.train.sparsity, 'k'); hold on;
  plot(1:epoch, info.train.saturation, 'k--');
  plot(1:epoch, info.val.sparsity, 'b'); 
  plot(1:epoch, info.val.saturation, 'b--');
  h=legend('train-sp', 'train-sat', 'val-sp', 'val-sat');
  grid on;
  xlabel('training epoch'); ylabel('measurements');
  set(h,'color','none');
  title('sparsity/saturation-epoch curve');


end


end
