function [M] = gen_basis_WMproj(Aa,PdS)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to approximate g(Ar+Bx+d) using a Taylor series expansion of the
% function g, which is assumed to be tanh
%
% Inputs
% Aa:   Coefficients for the polynomial basis representation of A*r+B*x+d
% PdS:  p x p matrix of shifts that are products of polynomials in Pd1
%
% Outputs
% M:    N x p matrix of the polynomial basis representation of g(Ar+Bx+d)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D    = tanh_deriv(Aa(:,1),5);
Ac   = Aa(:,2:end);
PdSr = PdS(2:end,2:end)-1;

Ac2 = zeros(size(Ac));
Ac3 = zeros(size(Ac));
Ac4 = zeros(size(Ac));
for i = 1:size(Ac,2)
    [Fi,Fj] = find(PdSr==i);
    Ac2(:,i) = sum(Ac(:,Fi)  .* Ac(:,Fj), 2);
end
for i = 1:size(Ac,2)
    [Fi,Fj] = find(PdSr==i);
    Ac3(:,i) = sum(Ac2(:,Fi) .* Ac(:,Fj), 2);
end
for i = 1:size(Ac,2)
    [Fi,Fj] = find(PdSr==i);
    Ac4(:,i) = sum(Ac3(:,Fi) .* Ac(:,Fj), 2);
end

M = [D(:,1), D(:,2).*Ac + D(:,3).*Ac2/2 + D(:,4).*Ac3/6 + D(:,5).*Ac4/24];

end