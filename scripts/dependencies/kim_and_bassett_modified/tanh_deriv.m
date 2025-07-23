function [D] = tanh_deriv(d,n)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% d: Vector of input
% n: order
%
% Outputs:
% D: Matrix of outputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
syms z; assume(z,'real');
D = tanh(z);
for i = 2:n
    D(:,i) = diff(D(:,i-1));
end
Df = matlabFunction(D);
D = Df(d);
end