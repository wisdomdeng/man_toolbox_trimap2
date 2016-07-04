function y = vl_nnfmapnoise_ff(x, noise, R)

sizeX = size(x);
if isa(x,'gpuArray'),
%  y = x + gpuArray(single(R).* randn(sizeX, 'single'));
   y = x .* bsxfun(@plus, 1.0, gpuArray(R.*noise));
else
%  y = x + single(R).* randn(sizeX, 'single');
   y = x .* bsxfun(@plus, 1.0, single(R.*noise));
end

end

