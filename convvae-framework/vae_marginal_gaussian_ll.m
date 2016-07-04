function mll = vae_marginal_gaussian_ll(x, z, enc_m, enc_s, dec_m, dec_s)
%% x:     Wx * Hx * Cx * N
%% z:     Wz * Hz * Cz * N
%% enc_m: Wx * Hx * Cx * N
%% enc_s: Wx * Hx * Cx * N
%% dec_m: Wz * Hz * Cz * N
%% dec_s: Wz * Hz * Cz * N
%% nsample = 1

dimx = numel(x) / size(x,4);
dimz = numel(z) / size(z,4);

%enc_s = min(max(enc_s, 1e-1), 1-1e-1);
%dec_s = min(max(dec_s, 1e-1), 1-1e-1);

numerator = -0.5*dimx*log(2*pi);
numerator = numerator + sum(sum(sum(-log(dec_s), 1), 2), 3);
numerator = numerator - 0.5 * sum(sum(sum(bsxfun(@rdivide, bsxfun(@minus, x, dec_m).^2, dec_s.^2), 1), 2), 3);
numerator = numerator - 0.5*dimz*log(2*pi);
numerator = numerator - 0.5 * sum(sum(sum(z.^2, 1), 2), 3);

denominator = -0.5*dimz*log(2*pi);
denominator = denominator + sum(sum(sum(-log(enc_s), 1), 2), 3);
denominator = denominator - 0.5 * sum(sum(sum(bsxfun(@rdivide, bsxfun(@minus, z, enc_m), enc_s).^2, 1), 2), 3);

%numerator(:)'
%denominator(:)'
%if nsample ~= 1
%  fprintf('Numerical Issue: nsample is gather than 1\n');
%end

mll = double(bsxfun(@minus, numerator, denominator));
%mll = exp(double(bsxfun(@minus, numerator, denominator)));
%mll = sum(mll(:))./nsample;
%mll = log(mll);

end
