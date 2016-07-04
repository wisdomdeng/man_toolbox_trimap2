% log-likelihood of gaussian latent variables

function logpz = compute_gaussian_loss(data, x, s, nsample)

nsample = numel(x)/numel(data);

% loss for real input
d = numel(x)/size(data, 4)/nsample;
x = reshape(x, [size(data), nsample]);
s = reshape(s, [size(data), nsample]);

logpz = 0.5*d*log(2*pi);
logpz = logpz + sum(sum(sum(log(s), 1), 2), 3);
logpz = (logpz + 0.5*sum(sum(sum(bsxfun(@rdivide, bsxfun(@minus, data, x).^2, s.^2), 1), 2), 3));
logpz = -squeeze(logpz); % 1 x nsample

return;
