function [GAM,DX,DDX,DDDX,DXDX,DXDXDX,DDXDX] = sym2deriv(dx,x,pr,Pd1,PdS)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% dx:  Symbolic equations of motion
% x:   Symbolic base variables
% pr:  List of prime numbers to uniquely identify symbols
% Pd1: p x k matrix of p polynomial basis terms as powers of k inputs
% Pds: shift matrix
%
% Outputs:
% ddx:          Second derivative
% dddx:         Third derivative
% ddxx:         First and second derivatives
% dxx:          First derivative squared
% dxxx:         First derivative cubed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Prefactors
pri = prod((pr').^Pd1,2);

% Obtain coefficients for the first derivative
ninputs  = numel(x);
noutputs = numel(dx);
nprimes  = numel(pri);

DX = zeros(noutputs,nprimes);
for i = 1:noutputs

    % Extract each term and coefficient in dx
    [a,b] = coeffs(dx(i)); a = double(a);

    % Numerically fast way to identify each term
    bs = double(subs(b,x,pr));

    % [ai1,~] = find(bs-pri==0);
    [~,ai] = min(abs(bs - pri));
    DX(i,ai) = a;
end

% Obtain the full map
GAM    = zeros(nprimes);
for i  = 2:nprimes
    p  = Pd1(i,:); 
    pi = find(p);
    for j = 1:length(pi)
        ps = p; 
        ps(pi(j)) = ps(pi(j))-1; 
        pd = PdS(pri-prod(pr'.^ps)==0,:); pdi = find(pd); pd = pd(pdi);
        GAM(i,pd) = GAM(i,pd) + p(pi(j))*DX(pi(j),pdi);
    end
end

% Second derivative
DDX = DX*GAM;

% Third derivative
DDDX = DDX*GAM;

% Prepare product terms
pd1i = find(sum(Pd1,2)==1); Pd1p = Pd1(pd1i,:);
pd2i = find(sum(Pd1,2)==2); Pd2p = Pd1(pd2i,:);
pd3i = find(sum(Pd1,2)==3); Pd3p = Pd1(pd3i,:);

% First derivative squared
ppi = zeros(length(pd2i),2);
DXDX = zeros(length(pd2i),size(Pd1,1));
for i = 1:length(pd2i)
    p = Pd1(pd2i(i),:); pf = zeros(1,length(p));
    for j = 1:1
        f = find(p,1); pf(f) = pf(f)+1; p(f) = p(f)-1;
    end
    ppi(i,:) = [find(prod(Pd1p==p,2)) find(prod(Pd1p==pf,2))];
end




for i = 1:size(Pd1,1)
    pd = PdS(i,:); pdi = find(pd); pd = pd(pdi);
    DXDX(:,pd) = DXDX(:,pd) + DX(ppi(:,1),pdi).*DX(ppi(:,2),i);
end

% First derivative cubed
ppi = zeros(length(pd3i),2);
DXDXDX = zeros(length(pd3i),size(Pd1,1));
for i = 1:length(pd3i)
    p = Pd1(pd3i(i),:); pf = zeros(1,length(p));
    for j = 1:1
        f = find(p,1); pf(f) = pf(f)+1; p(f) = p(f)-1;
    end
    ppi(i,:) = [find(prod(Pd2p==p,2)) find(prod(Pd1p==pf,2))];
end
for i = 1:size(Pd1,1)
    pd = PdS(i,:); pdi = find(pd); pd = pd(pdi);
    DXDXDX(:,pd) = DXDXDX(:,pd) + DXDX(ppi(:,1),pdi).*DX(ppi(:,2),i);
end


% First derivative cubed
ppi = zeros(length(pd2i),2);
ppic = ones(length(pd2i),1);
DDXDX = zeros(length(pd2i),size(Pd1,1));
for i = 1:length(pd2i)
    p = Pd1(pd2i(i),:); pf = zeros(1,length(p));
    for j = 1:1
        f = find(p,1); pf(f) = pf(f)+1; p(f) = p(f)-1;
    end
    ppi(i,:) = [find(prod(Pd1p==p,2)) find(prod(Pd1p==pf,2))];
    if(ppi(i,1)==ppi(i,2)); ppic(i) = 0.5; end
end
for i = 1:size(Pd1,1)
    pd = PdS(i,:); pdi = find(pd); pd = pd(pdi);
    DDXDX(:,pd) = DDXDX(:,pd) + ppic.*(DDX(ppi(:,1),pdi).*DX(ppi(:,2),i) + DDX(ppi(:,2),pdi).*DX(ppi(:,1),i));
end
end