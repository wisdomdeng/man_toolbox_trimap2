function spec = matconvnet_createSpec(net)

%
n = numel(net.layers);
spec = struct;
spec.inv_table = containers.Map();

%
for i = 1:n,
  %fprintf('%d\t[%s]\n', i, net.layers{i}.name);
  spec.inv_table(net.layers{i}.name) = i;
end

%% topological sorting
count = zeros(n, 1);
for i = 1:n, 
  %net.layers{i}
  for j = 1:numel(net.layers{i}.out),
   % fprintf('%d\t%d\t%s\n',i, j, net.layers{i}.out{j});
    
    indx = getIndx(net.layers{i}.out{j}, spec);
    count(indx) = count(indx) + 1;
  end
end

spec.queue = [];
pioneer = find(count == 0);
while ~isempty(pioneer) > 0
  spec.queue = [spec.queue, pioneer(1)];
  count(pioneer(1)) = -1;
  for j = 1:numel(net.layers{pioneer(1)}.out),
    indx = getIndx(net.layers{pioneer(1)}.out{j}, spec);
    count(indx) = count(indx) - 1;
  end
  pioneer = find(count == 0);
end

end

function indx = getIndx(name, spec)
  indx = spec.inv_table(name);
end
