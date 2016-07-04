function [varargout] = matconvnet_nntensorprod(xu, xv, w, b, dzdy)
%% Doc TODO
%% y = f(xu, xv, w, b)
%% xu: 1 * 1 * N * #sample
%% xv: 1 * 1 * M * #sample
%% w: 1 * 1 * (N + M + K) * F
%% b: 1 * K

%% [dzdu, dzdv, dzdw_u, dzdw_v, dzdw_y, dzdb_y] = f(xu, xv, w, b, dzdy)

% setup
N = size(gather(xu), 3);
M = size(gather(xv), 3);
K = size(gather(b), 2);
F = size(gather(w), 4);
nsample = size(gather(xu), 4);

if size(gather(w), 3) ~= N + M + K,
  fprintf('Dimensions do not match\n');
end

% reshaping
w_u = reshape(w(:,:,1:N,:), [N F]);
w_v = reshape(w(:,:,N+1:N+M,:), [M F]);
w_y = reshape(w(:,:,N+M+1:end,:), [K F]);
xu = reshape(xu, [N nsample]);
xv = reshape(xv, [M nsample]);
b = reshape(b, [K 1]);

%% forward & backward
if nargin <= 4 || isempty(dzdy)
  y = bsxfun(@plus, w_y * ((w_u'*xu).*(w_v'*xv)), b);
  y = reshape(y, [1 1 K nsample]);
  varargout = cell(1,1);
  varargout{1} = y;

else
  % K * nsample
  dzdy = reshape(dzdy, [K nsample]);

  % N * nsample
  dzdu = w_u * ((dzdy'*w_y).*(xv'*w_v))';
  dzdu = reshape(dzdu, [1 1 N nsample]);
  
  dzdv = w_v * ((dzdy'*w_y).*(xu'*w_u))';
  dzdv = reshape(dzdv, [1 1 M nsample]);
  
  dzdw_u = xu * ((dzdy'*w_y).*(xv'*w_v));
  dzdw_v = xv * ((dzdy'*w_y).*(xu'*w_u));
  dzdw_y = dzdy * ((xu'*w_u).*(xv'*w_v));
  dzdw = cat(1, dzdw_u, dzdw_v);
  dzdw = cat(1, dzdw, dzdw_y);
  dzdw = reshape(dzdw, [1 1 N+M+K F]);
  
  dzdb_y = sum(dzdy, 2);
  dzdb_y = reshape(dzdb_y, [1 K]);

  varargout = cell(1, 4);
  varargout{1} = dzdu;
  varargout{2} = dzdv;
  varargout{3} = dzdw;
  varargout{4} = dzdb_y;
end

end
