function W = getW(dx,x,basernn)

weights  = basernn.weights;
map      = basernn.bases_map;
map_shft = basernn.shift_matrix;
gam      = basernn.gam;

[nbases , noutputs] = size(map);

m = noutputs;
k = nbases;

% Convert symbolic output to source code by extracting coefficients
pr        = primes(2000)'; pr = pr(1:m);
[~,DXC]   = sym2deriv(dx,x,pr,map,map_shft);
o         = zeros(m,k); o(:,(1:m)+1) = eye(m);
oS        = DXC;
OdNPL     = o+oS/gam;

% Compile
W = lsqminnorm(weights', OdNPL')';

end

