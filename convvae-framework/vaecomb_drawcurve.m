function [info, opts] = vaecomb_drawcurve(info, opts, epoch)

  %% visualization
  % save
  info.train.objective(end) = info.train.objective(end) / info.numtrain;
  %info.train.lb(end) = info.train.lb(end) / info.numtrain;
  info.train.kl(end) = info.train.kl(end) / info.numtrain;
  info.train.ll(end) = info.train.ll(end) / info.numtrain;
  info.train.gc(end) = info.train.gc(end) / info.numtrain;
  info.train.speed(end) = info.numtrain / info.train.speed(end);
  
  info.val.objective(end) = info.val.objective(end) / info.numval;
  %info.val.lb(end) = info.val.lb(end) / info.numval;
  info.val.kl(end) = info.val.kl(end) / info.numval;
  info.val.ll(end) = info.val.ll(end) / info.numval;
  info.val.gc(end) = info.val.gc(end) / info.numval;
  info.val.speed(end) = info.numval / info.val.speed(end);

  if ~opts.verbose,
    fprintf(' train_speed = (%.1f images/s)\t val_speed = (%.1f images/s)\n', ...
       info.train.speed(end), info.val.speed(end));
    fprintf(' train_obj = (%.3f)\t val_lb = (%.3f)\n ', ...
       info.train.objective(end), info.val.objective(end));
    fprintf(' train_kl = (%.3f)\t val_kl = (%.3f)\n ', ...
       info.train.kl(end), info.val.kl(end));
    fprintf(' train_ll = (%.3f)\t val_ll = (%.3f)\n ', ...
       info.train.ll(end), info.val.ll(end));
    fprintf(' train_gc = (%.3f)\t val_gc = (%.3f)\n ', ...
       info.train.gc(end), info.val.gc(end));
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
    semilogy(1:epoch, info.train.kl, 'k') ; hold on;
    semilogy(1:epoch, info.val.kl, 'b') ;
    h=legend('train','val') ;
    grid on ;
    xlabel('training epoch') ; ylabel('kl') ;
    set(h,'color','none') ;
    title('KL') ;


    subplot(2,2,3);
    semilogy(1:epoch, info.train.ll, 'k') ; hold on;
    semilogy(1:epoch, info.val.ll, 'b') ;
    h=legend('train','val') ;
    grid on;
    xlabel('training epoch'); ylabel('ll') ;
    set(h, 'color', 'none') ;
    title('LogLikelihood');

  end

end
