function corrupt_x = apply_corruption(x, cpt_mode)

corrupt_x = x;
switch cpt_mode,
case 1 %% zero-masking 25 %
  corrupt_x(rand(size(x))<0.25) = 0;
case 2 %% zero-masking 50 %
  corrupt_x(rand(size(x))<0.5) = 0;
%%
case 11 %% salt-and-pepeer noise 25%
  loc = rand(size(x))<0.25;
  coin_flip = rand(size(x))>0.5;
  corrupt_x = bsxfun(@times, x, 1-loc) + bsxfun(@times, loc, coin_flip);
case 12 %% salt-and-pepper noise 50%
  loc = rand(size(x))<0.5;
  coin_flip = rand(size(x))>0.5;
  corrupt_x = bsxfun(@times, x, 1-loc) + bsxfun(@times, loc, coin_flip);
case 21 %% additive gaussian noise 25%
  corrupt_x = bsxfun(@plus, x, 0.25*randn(size(x)));
case 22 %% additive gaussian noise 50%
  corrupt_x = bsxfun(@plus, x, 0.5*randn(size(x)));
case 23 %% additive gaussian noise 10%
  corrupt_x = bsxfun(@plus, x, 0.1*randn(size(x)));
end %% end of switch

end
