function basernn = program_rnn(runParams, board_params)
%%
disp('Making RNN')


[x,dx] = define_symbolic_logic(board_params);

n     = runParams.n;
dt    = runParams.dt;

m  = numel(dx); % number of outputs for the base RNN

% Decompile base RNN into basis set
basernn = getexpansion(n , m, dt);

% Uses symbolic programmed logic to compile W for base RNN
if strcmpi(runParams.connectivity,"random")
    A = (rand(n)-.5).*(rand(n)<.05);                     % Recurrent matrix
    A = sparse(A / abs(eigs(A,1,'largestabs','MaxIterations',1e6))) * .01;   % Normalize
    basernn.A = A;
end

W = solve_readout(dx,x,basernn);

basernn.W = W;


end

