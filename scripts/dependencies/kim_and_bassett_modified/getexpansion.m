function [basernn] = getexpansion(n_neurons , n_outputs, dt)
% Time
gam = 100;

% Reservoir
% Define reservoir
A  = sparse(zeros(n_neurons));
B  = (rand(n_neurons,n_outputs)-.5)*.1;
rs = (rand(n_neurons,1)-.5);
xs = zeros(n_outputs,1);

% Init base RNN with randomized weights
basernn = ReservoirTanhB_dc(A,B,rs,xs,dt,gam); d = basernn.d;

% Fixed points for new shift
basernn.r = rs; rsT = rs;

% Decompile (Get the basis set)
dv       = A*rsT + B*xs + d;
[Pd1,C1] = decomp_poly1_ns(A,B,rsT,dv,4);

% Compute shift matrix
[Pdx,Pdy] = find((sum(Pd1,2)+sum(Pd1,2)') <= max(sum(Pd1,2)));
PdS       = zeros(size(Pd1,1));

for i = 1:length(Pdx)
    PdS(Pdx(i),Pdy(i)) = find(sum(Pd1==(Pd1(Pdx(i),:)+Pd1(Pdy(i),:)),2)==n_outputs);
end

% Finish generating basis
Aa = zeros(size(C1));
Aa(:,(1:n_outputs)+1)  = Aa(:,(1:n_outputs)+1)+B;  Aa(:,1) = Aa(:,1) + d;
GdNPL = gen_basis(Aa,PdS);


basernn.bases_map    = Pd1;
basernn.shift_matrix = PdS;
basernn.weights      = GdNPL;

end

