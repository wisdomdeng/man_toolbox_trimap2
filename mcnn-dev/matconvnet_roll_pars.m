%% Last Update: March. 10, 2015 by Xinchen Yan (add tensorprod layer)
function theta = matconvnet_roll_pars(net)

theta = [];
for i = 1:numel(net.layers)
  switch net.layers{i}.type
  case {'conv','tensorprod'}
    theta_filters = gather(net.layers{i}.filters);
    theta_biases = gather(net.layers{i}.biases);
    theta = [theta; theta_filters(:); theta_biases(:)];
  case 'bias'
    theta_biases = gather(net.layers{i}.biases);
    theta = [theta; theta_biases(:)];
  end
end

end
