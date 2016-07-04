function [cost, res_rec, res_gen] = cvae_cost(x, style, noise, net_rec, net_gen, res_rec, res_gen)

gpuMode = isa(x, 'gpuArray');
one = single(1);
if gpuMode,
    one = gpuArray(one);
end

cost = 0;

% -------------------------------------------------------------------------
%                                                             recognition
% -------------------------------------------------------------------------


res_rec = cvae_ff(net_rec, x, [], style, noise, res_rec);
cost = cost + res_rec(end).x;

%size(res_rec(end-1).x)
%size(res_rec(end-1).s)
% -------------------------------------------------------------------------
%                                                              generation
% -------------------------------------------------------------------------

net_gen.layers{end}.data = x;
res_gen = cvae_ff(net_gen, res_rec(end-1).x, res_rec(end-1).s, style, noise, res_gen);
cost = cost + res_gen(end).x;


% -------------------------------------------------------------------------
%                                                   generation (gradient)
% -------------------------------------------------------------------------

res_gen = cvae_bp(net_gen, style, noise, one, res_gen);


% -------------------------------------------------------------------------
%                                                  recognition (gradient)
% -------------------------------------------------------------------------

res_rec(end).dzdx = res_gen(1).dzdx;
res_rec(end).dzds = res_gen(1).dzds;
res_rec = cvae_bp(net_rec, style, noise, [], res_rec);


return;
