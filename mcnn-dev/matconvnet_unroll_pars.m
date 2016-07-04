% Last Update: March. 10, 2015 by Xinchen Yan (add tensorprod layer)
function [net] = matconvnet_unroll_pars(theta, net, useGpu)
%size(theta)
curHead = 1;

for i = 1:numel(net.layers),
  switch net.layers{i}.type
  case {'conv','tensorprod'}

    if useGpu,
      curL = length(gather(net.layers{i}.filters(:)));
      net.layers{i}.filters = reshape(gpuArray(single(theta(curHead:curHead+curL-1))), size(net.layers{i}.filters));
      curHead = curHead + curL;
    
      curL = length(gather(net.layers{i}.biases(:)));
      net.layers{i}.biases = reshape(gpuArray(single(theta(curHead:curHead+curL-1))), size(net.layers{i}.biases));
      curHead = curHead + curL;

    else
      curL = length(net.layers{i}.filters(:));
      net.layers{i}.filters = reshape(theta(curHead:curHead+curL-1), size(net.layers{i}.filters)); 
      curHead = curHead + curL;
    
      curL = length(net.layers{i}.biases(:));
      net.layers{i}.biases = reshape(theta(curHead:curHead+curL-1), size(net.layers{i}.biases));
      curHead = curHead + curL;
    end
  
  case 'bias'
    
    if useGpu,
      curL = length(gather(net.layers{i}.biases(:)));
      net.layers{i}.biases = reshape(gpuArray(single(theta(curHead:curHead+curL-1))), size(net.layers{i}.biases));
      curHead = curHead + curL;
    else
      curL = length(net.layers{i}.biases(:));
      net.layers{i}.biases = reshape(theta(curHead:curHead+curL-1), size(net.layers{i}.biases));
      curHead = curHead + curL;
    end

end

end
