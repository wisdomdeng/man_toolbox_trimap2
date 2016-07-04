function dzdx = vl_nnfmapnoise_bp(x, dzdy, noise, R)

if isa(x, 'gpuArray'),
  dzdx = dzdy .* bsxfun(@plus, 1.0, gpuArray(R.*noise));
else
  dzdx = dzdy .* bsxfun(@plus, 1.0, single(R.*noise));
end

end
