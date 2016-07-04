function [cost, res_rec, res_gen] = vae_cost(x, noise, net_rec, net_gen, res_rec, res_gen, cpt_mode)

if ~exist('cpt_mode') || isempty(cpt_mode),
  cpt_mode = 0;
end

gpuMode = isa(x, 'gpuArray');
one = single(1);
if gpuMode,
    one = gpuArray(one);
end

cost = 0;

% -------------------------------------------------------------------------
%                                                             recognition
% -------------------------------------------------------------------------
if cpt_mode == 0,
  res_rec = vae_ff(net_rec, x, [], noise, res_rec);
else
  corrupt_x = apply_corruption(x, cpt_mode);
  res_rec = vae_ff(net_rec, corrupt_x, [], noise, res_rec);
end
cost = cost + res_rec(end).x;

%size(res_rec(end-1).x)
%size(res_rec(end-1).s)
% -------------------------------------------------------------------------
%                                                              generation
% -------------------------------------------------------------------------

net_gen.layers{end}.data = x;
res_gen = vae_ff(net_gen, res_rec(end-1).x, res_rec(end-1).s, noise, res_gen);
cost = cost + res_gen(end).x;


% -------------------------------------------------------------------------
%                                                   generation (gradient)
% -------------------------------------------------------------------------

res_gen = vae_bp(net_gen, noise, one, res_gen);


% -------------------------------------------------------------------------
%                                                  recognition (gradient)
% -------------------------------------------------------------------------

res_rec(end).dzdx = res_gen(1).dzdx;
res_rec(end).dzds = res_gen(1).dzds;
res_rec = vae_bp(net_rec, noise, [], res_rec);


return;
