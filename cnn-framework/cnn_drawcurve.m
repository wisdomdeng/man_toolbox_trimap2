function [info, opts] = cnn_drawcurve(info, opts, epoch)

  %% visualization
  % save
  info.train.objective(end) = info.train.objective(end) / info.numtrain;
  info.train.error(end) = info.train.error(end) / info.numtrain;
  info.train.speed(end) = info.numtrain / info.train.speed(end);
  
  info.val.objective(end) = info.val.objective(end) / info.numval;
  info.val.error(end) = info.val.error(end) / info.numval;
  info.val.speed(end) = info.numval / info.val.speed(end);

  if strcmp(opts.errorType,'binary'),
    info.train.ap(end) = info.train.ap(end) / info.numtrain;
    info.train.iou(end) = info.train.iou(end) / info.numtrain;
    
    info.val.ap(end) = info.val.ap(end) / info.numval;
    info.val.iou(end) = info.val.iou(end) / info.numval;
  elseif strcmp(opts.errorType, 'multiclass'),
    info.train.cls_err(end) = info.train.cls_err(end) / info.numtrain;
    info.val.cls_err(end) = info.val.cls_err(end) / info.numval;

  end

  if ~opts.verbose,
    fprintf(' train_speed = (%.1f images/s)\t val_speed = (%.1f images/s)\n', ...
       info.train.speed(end), info.val.speed(end));
    fprintf(' train_error = (%.3f)\t val_error = (%.3f)\n ', ...
       info.train.error(end), info.val.error(end));
    if strcmp(opts.errorType, 'binary'),
       fprintf(' train_ap = (%.3f)\t val_ap = (%.3f)\n', ...
         info.train.ap(end), info.val.ap(end));
       fprintf(' train_iou = (%.3f)\t val_iou = (%.3f)\n', ...
         info.train.iou(end), info.val.iou(end));
    end
    fprintf('\n');
  end

  if opts.verbose,
    figure(1) ; clf ;
    subplot(2,2,1) ;
    semilogy(1:epoch, info.train.objective, 'k') ; hold on ;
    semilogy(1:epoch, info.val.objective, 'b') ;
    xlabel('training epoch') ; ylabel('energy') ;
    grid on ;
    h=legend('train', 'val') ;
    set(h,'color','none');
    title('objective') ;

    subplot(2,2,2) ;
    switch opts.errorType
      case 'multiclass'
        plot(1:epoch, info.train.error, 'k') ; hold on ;
        plot(1:epoch, info.train.cls_err, 'k--') ;
        plot(1:epoch, info.val.error, 'b') ;
        plot(1:epoch, info.val.cls_err, 'b--') ;
        h=legend('train','train-cls_err','val','val-cls_err') ;
      case 'binary'
        plot(1:epoch, info.train.error, 'k') ; hold on ;
        plot(1:epoch, info.val.error, 'b') ;
        h=legend('train','val') ;
    end
    grid on ;
    xlabel('training epoch') ; ylabel('error') ;
    set(h,'color','none') ;
    title('error') ;


    subplot(2,2,3);
    switch opts.errorType
      case 'multiclass'
      case 'binary'
        plot(1:epoch, info.train.ap, 'k') ; hold on;
        plot(1:epoch, info.train.iou, 'k--');
        plot(1:epoch, info.val.ap, 'b') ;
        plot(1:epoch, info.val.iou, 'b--');
        h=legend('train-ap','train-iou','val-ap','val-iou') ;
    end

    grid on;
    xlabel('training epoch'); ylabel('score') ;
    set(h, 'color', 'none') ;
    title('evaluation metrics');


    %drawnow;
    %print(1, modelFigPath, '-dpdf');
  end

end
