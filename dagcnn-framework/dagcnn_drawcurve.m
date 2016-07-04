function [info, opts] = dagcnn_drawcurve(info, opts, epoch)

  %% visualization
  % save
  info.train.objective(end) = info.train.objective(end) / info.numtrain;
  info.train.error(end) = info.train.error(end) / info.numtrain;
  info.train.speed(end) = info.numtrain / info.train.speed(end);
  
  info.val.objective(end) = info.val.objective(end) / info.numval;
  info.val.error(end) = info.val.error(end) / info.numval;
  info.val.speed(end) = info.numval / info.val.speed(end);

  switch opts.errorType
  case 'binary'
    info.train.ap(end) = info.train.ap(end) / info.numtrain;
    info.train.iou(end) = info.train.iou(end) / info.numtrain;
    
    info.val.ap(end) = info.val.ap(end) / info.numval;
    info.val.iou(end) = info.val.iou(end) / info.numval;
  case 'multiclass'
    %info.train.topFiveError(end) = info.train.topFiveError(end) / info.numtrain;

    %info.val.topFiveError(end) = info.val.topFiveError(end) / info.numval;
    info.train.class_error(end) = info.train.class_error(end) / info.numtrain;

    info.val.class_error(end) = info.val.class_error(end) / info.numval;
  case 'recon'
    info.train.sat(end) = info.train.sat(end) / info.numtrain;
    info.train.sat2(end) = info.train.sat2(end) / info.numtrain;

    info.val.sat(end) = info.val.sat(end) / info.numval;
    info.val.sat2(end) = info.val.sat2(end) / info.numval;
  end

  if ~opts.verbose,
    fprintf(' train_speed = (%.1f images/s)\t val_speed = (%.1f images/s)\n', ...
       info.train.speed(end), info.val.speed(end));
    fprintf(' train_error = (%.3f)\t val_error = (%.3f)\n ', ...
       info.train.error(end), info.val.error(end));
    switch opts.errorType
    case 'binary'
       fprintf(' train_ap = (%.3f)\t val_ap = (%.3f)\n', ...
         info.train.ap(end), info.val.ap(end));
       fprintf(' train_iou = (%.3f)\t val_iou = (%.3f)\n', ...
         info.train.iou(end), info.val.iou(end));
    case 'multiclass'
      fprintf(' train_cls_err = (%.3f)\t val_cls_err = (%.3f)\n', ...
        info.train.class_error(end), info.val.class_error(end));

    case 'recon'
       fprintf(' train_sat = (%.3f)\t val_sat = (%.3f)\n', ...
         info.train.sat(end)*100, info.val.sat(end)*100);
       fprintf(' train_sat2 = (%.3f)\t val_sat2 = (%.3f)\n', ...
         info.train.sat2(end)*100, info.val.sat2(end)*100);
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
        %plot(1:epoch, info.train.topFiveError, 'k--') ;
        plot(1:epoch, info.val.error, 'b') ;
        %plot(1:epoch, info.val.topFiveError, 'b--') ;
        h=legend('train','val') ;
      case 'binary'
        plot(1:epoch, info.train.error, 'k') ; hold on ;
        plot(1:epoch, info.val.error, 'b') ;
        h=legend('train','val') ;
      case 'recon'
        plot(1:epoch, info.train.error, 'k') ; hold on;
        plot(1:epoch, info.val.error, 'b') ;
        h=legend('train','val') ;
    end
    grid on ;
    xlabel('training epoch') ; ylabel('error') ;
    set(h,'color','none') ;
    title('error') ;


    subplot(2,2,3);
    switch opts.errorType
      case 'recon'
        plot(1:epoch, info.train.sat, 'k') ; hold on;
        plot(1:epoch, info.train.sat2, 'k--') ;
        plot(1:epoch, info.val.sat, 'b') ;
        plot(1:epoch, info.val.sat2, 'b--') ;
        h = legend('train-sat','train-sat2','val-sat','val-sat2');
      case 'multiclass'
        plot(1:epoch, info.train.class_error,'k') ; hold on ;
        plot(1:epoch, info.val.class_error,'b') ; hold on ;
        h = legend('train-cls-err','val-cls-err');
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

  end

end
