function [] = visualize_hidden_units( group_idx )
%VISUALIZE_HIDDEN_UNITS Summary of this function goes here
%   Detailed explanation goes here
alpha = 0.1;
fname = sprintf('models/aorta_split_%02d_%s_itr_%d_al%g.mat', group_idx, 'lbfgs', 200, alpha);
%fname
load(fname);
vishid = reshape(weights.vishid, params.ws^2*params.numch, params.numhid);
fig = figure(1);
display_network_nonsquare(vishid);

print(sprintf('vis/hidden_units/aorta_split_%02d_%s_itr_%d_al%g.png', group_idx, 'lbfgs', 200, alpha), fig, '-dpng');

end

