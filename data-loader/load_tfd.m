% ==========================================
% load toronto face database
% imsize = 32, 48
%
% xlab = 2304 x numdata (e.g., imsize = 48)
% ==========================================


function [xlab, ylab_ex, ylab_id, folds, xunlab, dim] = load_tfd(imsize)

if ~exist('imsize', 'var'),
    imsize = 48;
end

if imsize == 32,
    TFD = load('/mnt/neocortex/scratch/kihyuks/libdeepnets/trunk/pmRBM/nips_ws/pmrbm_tfd/TFD_32x32.mat');
    dim = [32 32 1];
elseif imsize == 48,
    TFD = load('~/neo4/vae-test/data/TFD_48x48.mat');
    dim = [48 48 1];
else
    error('image size should be either 32 or 48');
end


% labeled data
labeled_idx = TFD.labs_ex ~= -1;
xlab = TFD.images(labeled_idx, :, :);
xlab = im2double(xlab);
xlab = reshape(xlab, size(xlab, 1), size(xlab, 2)*size(xlab, 3))';
ylab_ex = TFD.labs_ex(labeled_idx);
ylab_id = TFD.labs_id(labeled_idx);
folds = TFD.folds(labeled_idx, :);

% unlabeled data
unlab_idx = TFD.labs_ex == -1;
xunlab = TFD.images(unlab_idx, :, :);
xunlab = im2double(xunlab);
xunlab = reshape(xunlab, size(xunlab, 1), size(xunlab, 2)*size(xunlab, 3))';
xunlab(:, sum(xunlab.^2, 1) == 0) = []; % remove examples with all 0's


return;
