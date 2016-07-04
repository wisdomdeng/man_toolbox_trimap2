function numgrad = computeNumericalGradient( J, theta )
%COMPUTENUMERICALGRADIENT Summary of this function goes here
%   Detailed explanation goes here

numgrad = zeros(size(theta));

eps = 1e-2;
curEps = zeros(size(theta));

for i = 1:size(theta,1),
    curEps(i) = eps;
    y1 = J(theta+curEps);
    y2 = J(theta-curEps);
    numgrad(i,1) = (y1-y2)./(2.0*eps);
    curEps(i) = 0.0;
end

end

