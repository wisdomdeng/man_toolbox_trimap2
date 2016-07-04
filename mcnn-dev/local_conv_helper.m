function [InRange, Padding, OutRange] = ...
  local_conv_helper(nGroup, Insize, Instep, fsize, pad)

for i = 1:nGroup,
  InRange(i,1) = (i-1)*Instep+1;
  InRange(i,2) = i*Instep;
  Padding(i,1:4) = [0 0 0 0];
  if (i==1), 
    Padding(i,1) = pad(1); 
    Padding(i,3) = pad(3); 
    InRange(i,2) = InRange(i,2) + (fsize-1)/2;
    OutRange(i,1) = 1;
  elseif (i==nGroup),
    Padding(i,2) = pad(2);
    Padding(i,4) = pad(4);
    InRange(i,1) = InRange(i,1) - (fsize-1)/2;
    InRange(i,2) = Insize;
    OutRange(i,1) = OutRange(i-1,2) + 1;
  else
    InRange(i,1) = InRange(i,1) - (fsize-1)/2;
    InRange(i,2) = InRange(i,2) + (fsize-1)/2;
    OutRange(i,1) = OutRange(i-1,2) + 1;
  end
  numPixels = (InRange(i,2) - InRange(i,1) + 1 - (fsize - 1) + Padding(i, 1) + Padding(i, 2));
  OutRange(i,2) = OutRange(i,1) + numPixels - 1;
end


end
