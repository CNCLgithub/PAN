function W = solve_readout(dx,x,basernn)

weights  = basernn.weights;
map      = basernn.bases_map;
map_shft = basernn.shift_matrix;
gam      = basernn.gam;

[nbases , noutputs] = size(map);
%%
nvars = symvar_check(dx);
%%
% Convert symbolic output to source code by extracting coefficients
pr        = primes(2000)'; pr = pr(1:noutputs);
[~,DXC]   = sym2deriv(dx,x(1:nvars),pr,map,map_shft);

o         = zeros(noutputs,nbases); 
o(:,(1:noutputs)+1) = eye(noutputs);

oS        = DXC;
OdNPL     = o+oS/gam;

% Compile
W = lsqminnorm(weights', OdNPL')';

end

%%

function nvars = symvar_check(dx)

% Convert the symbolic matrix to a string
dx_str = char(dx);

% Use regexp to find occurrences of the pattern 'x#(t)'
pattern = 'x\d+\(t\)';
matches = regexp(dx_str, pattern, 'match');

% Get the unique matches
unique_matches = unique(matches);

% Display the result
% disp('Unique symbolic variables in the matrix dx:');
% disp(unique_matches);

% Count the number of unique matches
nvars = length(unique_matches);
% disp('Number of unique symbolic variables:');
% disp(nvars);


end

