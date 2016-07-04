function mll = vae_marginal_bernoulli_ll(x, z, enc_m, enc_s, dec_m)
%% x:     Wx * Hx * Cx * 1
%% z:     Wz * Hz * Cz * nsample
%% enc_m: Wx * Hx * Cx * 1
%% enc_s: Wx * Hx * Cx * 1
%% dec_m: Wz * Hz * Cz * nsample

nsample = size(z, 4);
dimx = numel(x);
dimz = numel(z) / nsample;

numerator = sum(sum(sum(bsxfun(@times, x, log(dec_m)) + bsxfun(@times, 1-x, log(1-dec_m)), 1), 2), 3);
numerator = numerator - 0.5*dimz*log(2*pi);
numerator = numerator - 0.5 * sum(sum(sum(z.^2, 1), 2), 3);

denominator = -0.5*dimz*log(2*pi);
denominator = denominator + sum(sum(sum(-log(enc_s), 1), 2), 3);
denominator = denominator - 0.5 * sum(sum(sum(bsxfun(@rdivide, bsxfun(@minus, z, enc_m).^2, enc_s.^2), 1), 2), 3);

mll = exp(double(bsxfun(@minus, numerator, denominator)));
mll = sum(mll(:))./nsample;
mll = log(mll);

end
