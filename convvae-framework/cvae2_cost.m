function [cost, res_rec, res_gen, res_style1, res_style2] = cvae2_cost(x, style, noise, net_rec, net_gen, ...
  net_style1, net_style2, res_rec, res_gen, res_style1, res_style2)

gpuMode = isa(x, 'gpuArray');
one = single(1);
if gpuMode,
    one = gpuArray(one);
end

cost = 0;

% -------------------------------------------------------------------------
%                                                             recognition
% -------------------------------------------------------------------------

[res_rec, res_style1] = cvae2_ff(net_rec, net_style1, ...
	x, [], style, noise, res_rec, res_style1);
cost = cost + res_rec(end).x;

%size(res_rec(end-1).x)
%size(res_rec(end-1).s)
% -------------------------------------------------------------------------
%                                                              generation
% -------------------------------------------------------------------------

net_gen.layers{end}.data = x;
[res_gen, res_style2] = cvae2_ff(net_gen, net_style2, ...
  res_rec(end-1).x, res_rec(end-1).s, style, noise, res_gen, res_style2);
cost = cost + res_gen(end).x;


% -------------------------------------------------------------------------
%                                                   generation (gradient)
% -------------------------------------------------------------------------

[res_gen, res_style2] = cvae2_bp(net_gen, net_style2, style, ...
  noise, one, res_gen, res_style2);


% -------------------------------------------------------------------------
%                                                  recognition (gradient)
% -------------------------------------------------------------------------

res_rec(end).dzdx = res_gen(1).dzdx;
res_rec(end).dzds = res_gen(1).dzds;
[res_rec, res_style1] = cvae2_bp(net_rec, net_style1, style, ...
  noise, [], res_rec, res_style1);

return;
