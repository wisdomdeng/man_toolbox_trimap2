function elem = cpu2gpu_copy(elem, useGPU)

switch useGPU
case 0
  elem = single(elem);
case 1
  elem = gpuArray(single(elem));
case 2
  elem = gsingle(elem);
end

end
