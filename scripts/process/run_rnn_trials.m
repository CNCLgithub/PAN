function rnnvars = run_rnn_trials(basernn, trial_data, varargin)
%%

% Parse optional arguments
p = inputParser;
addParameter(p,"printStuff"  ,  true, @islogical)
addParameter(p,"binStates"   ,  true, @islogical)
addParameter(p,"padmov_trig" , false, @islogical)
addParameter(p,"input_data"  ,    [], @isnumeric)
addParameter(p,"parallelFlag",  true, @islogical)


parse(p, varargin{:});

input_data   = p.Results.input_data;
padmov_trig  = p.Results.padmov_trig;
printStuff   = p.Results.printStuff;
parallelFlag = p.Results.parallelFlag;

% Initial setup
x0      = trial_data.x0;
y0      = trial_data.y0;
xdot0   = trial_data.xdot0;
ydot0   = trial_data.ydot0;
runTime = trial_data.simulation_time;

if isscalar(runTime), repmat(runTime,numel(x0),1); end

% Load base rnn
dt = basernn.delT;
W  = basernn.W;

% Set up for recurrent and input indices
reccurent_idx = 1:12;
input_idx     = 13;

% Prepare the programmed (closed-loop/recurrent) PAN model
A   = basernn.A;
B   = basernn.B;
rs  = basernn.rs;
gam = basernn.gam;

recW = W(reccurent_idx, :);
BW = B(:, reccurent_idx) * recW;
Bx = B(:, input_idx);
xs = zeros(numel(input_idx), 1);
prnn = ReservoirTanhB_dc(A + BW, Bx, rs, xs, dt, gam);
prnn.d = basernn.d;

if padmov_trig
    plims = trial_data.board_params.paddle_var_limits;
else,plims=[0 0];
end


% Function handle for training/loading-in initial conditions
f = @(c) basernn.train(repmat(c, [1, 200, 4]));

% Prepare for parallel execution
ntrials = numel(x0);
numberOutputs = 4;
k = rank(W);

% Initialize containers
[network_states, start_states,network_inputs,states_init] =...
    deal(cell(ntrials, 1));

if parallelFlag
    
    futures = parallel.FevalFuture.empty(0, ntrials);
    pool = gcp('nocreate');
    if isempty(pool)
        pool = parpool;  % Default number of workers
    end
        
    % Dispatch trials to workers
    for i = 1:ntrials
        
        inputs = {prnn, x0(i), y0(i), xdot0(i), ydot0(i), runTime(i),...
            padmov_trig, f, plims, k,input_data};
        futures(i) = parfeval(pool, @run_single_trial, numberOutputs, ...
            inputs);
    end
       
    % Collect results
    for i = 1:ntrials
        if printStuff,disp([num2str(i) '/' num2str(ntrials)]),end
        
        [completedIdx,start,states,trial_inputs,burnin] = fetchNext(futures);
        
        start_states{completedIdx}    = start;
        network_states{completedIdx}  = states;
        network_inputs{completedIdx}  = trial_inputs;
        states_init{completedIdx}     = burnin;
        
    end
    
    p = gcp; % Get the current parallel pool
    parfevalOnAll(p,@clear,0,"all");

else

    for i = 1:ntrials
        disp(['   Trial ' num2str(i) '/' num2str(ntrials)])

        inputs = {prnn, x0(i), y0(i), xdot0(i), ydot0(i), runTime(i),...
            padmov_trig, f, plims, k,input_data};

        [start_states{i},network_states{i} ,...
        network_inputs{i},states_init{i}  ] = run_single_trial(inputs);

    end

end


% Package the results
rnnvars = prnnvars();

progrnn.basernn = basernn;
progrnn.prnn    = prnn;
progrnn.f       = f;
progrnn.k       = k;

rnnvars.board_params    = trial_data.board_params;
rnnvars.progrnn         = progrnn; 
rnnvars.data            = trial_data;
rnnvars.input_data      = network_inputs;
rnnvars.network_states  = network_states;
rnnvars.start_states    = start_states;
rnnvars.W               = W;

% Load Brain data containing the binned logicalmasks for time bins associated 
% with game play. Rishi et al. had 6 time bins of prep with ball on screen
% but no movement; there were 3 tiers of velocity magnitude (speeds) and
% thus depending on ||v||, 𝜃, and (x0,y0) all trials last different amount
% of time bins.
load(fullfile("../../data/brain_and_rishi_data.mat"),"brain");

bint = 50; % bin by 50ms like Rishi
mask = brain.all_mask;

startbin   = cellfun(@(v) find(v,1,"first"),num2cell(mask,2));
finalbin   = cellfun(@(v) find(v,1,"last"),num2cell(mask,2));
finalTime  = finalbin.*bint;

trial_mask.mask = mask;
trial_mask.startbin  = startbin;
trial_mask.finalbin  = finalbin;
trial_mask.finalTime = finalTime;

[normed_init_states, normed_trial_states]=norm_network_states(...
    network_states, ...
    states_init,finalTime);

rnnvars.initial_states_raw      = states_init;   
rnnvars.network_states_raw      = network_states;   
rnnvars.initial_states_normed   = normed_init_states;      
rnnvars.network_states_normed   = normed_trial_states;    

% bin and normalize continuous PAN model's state evolution to match the
% dimension and parameters of the neural data and the ML-style models
binnedStates_raw = bin_network_states_prnn( ...
    network_states, ...
    states_init, ...
    bint);

binnedStates_norm = bin_network_states_prnn( ...
    normed_trial_states, ...
    normed_init_states, ...
    bint);

all_states = cellfun(@(stcell,fncell) cat(2, stcell,fncell), ...
    normed_init_states,normed_trial_states,uniformoutput=0);

rnnvars.trial_mask            = trial_mask;
rnnvars.network_states        = all_states;
rnnvars.binned_states         = binnedStates_norm;
rnnvars.binned_states_raw     = binnedStates_raw;


end

function [start_states, network_states, network_inputs,states_init] = run_single_trial(inputs)

[prnn, x0, y0, xdot0, ydot0, runTime,...
padmov_trig, f, plims , k,input_data] = inputs{:};


xstart =    x0;
ystart =    y0;
xvel0  = xdot0;
yvel0  = ydot0;

ntpts  = runTime;

padl = [0 0]; % [startpos endpos]    

% Initialize burnin of base RNNs to get the appropriate starting state
% for these initial conditions
init_conds = [
    abs(xvel0) ; abs(yvel0)
    xstart ; ystart
    sign(xvel0)*0.1 ; -sign(xvel0)*0.1
    sign(yvel0)*0.1 ; -sign(yvel0)*0.1
    0;0;0;0;padl(1)
    ];

% Train the base RNN on these initial connditions to get RNN start
% state after "burn-in"
states_init = f(init_conds);
    
prnnt   = prnn;
prnnt.r = states_init(:,end);

start_states = states_init(:,end);

% Drive/Evolve with paddle inputs and play game
if padmov_trig
     pf = (rand*plims) - (plims/2);
     xt = linspace(0,pf,ntpts);
else,xt = linspace(padl(1),padl(2),ntpts);
end

if ~isempty(input_data), xt = input_data;end

network_state = prnnt.trainSVD(repmat(xt,[1,1,4]),k); 
% network_game = prnn.train(repmat(xt,[1,1,4])); 

network_states = network_state; 
network_inputs = xt;

end

function [normed_init_states, normed_trial_states]=norm_network_states(...
    network_states,states_init,finalTime)
%%
    % nfunc = @(s , rs) zscore(s - rs, [] , [2 3]);
    % nfunc = @(s , rs) zscore(s, [] , [2 3]);
    % states = rnnvars.normFunc(rnnvars);
    % nfunc  = @(s , rs) s - rs;

    ntrials = numel(network_states);
    normed_init_states  = cell(ntrials,1);
    normed_trial_states = cell(ntrials,1);


    for i = 1:ntrials

        
        submat = network_states{i}(:,1:finalTime(i));

        avgdata = mean(submat,2);
        stddata = std(submat,1,2);

        normed_init_states{i}  = (states_init{1}- avgdata)./stddata;
        normed_trial_states{i} = (network_states{i} - avgdata)./stddata;

    end

if 0
    imagesc(normed_trial_states{1})
    clim([-3,3])
end

end

function  modelState_bin = bin_network_states_prnn(sim_states,init_states,bint)

    % nfunc = @(s , rs) zscore(s - rs, [] , [2 3]);
    % nfunc = @(s , rs) zscore(s, [] , [2 3]);
    % states = rnnvars.normFunc(rnnvars);
    % nfunc  = @(s , rs) s - rs;

    states  = sim_states;
    ntrials = numel(states);

    modelStates= cell(ntrials,1);

    inittime = size(init_states{1},2)/bint;

    for i = 1:ntrials

        binnedInit = bin_state_fcn(init_states{i},  bint);
        binnedSim  = bin_state_fcn(sim_states{i},  bint);

        modelStates{i} = horzcat(binnedInit,binnedSim);

    end

    % Save full state matrices for the trials, after combining the initial
    % burn-in period and the actual simulation
    % 'burninTime' refers to the in
    modelState_bin.burninTime  = inittime;
    modelState_bin.modelStates = modelStates;
    
end

function binned = bin_state_fcn(state, bint)

    [rows, cols] = size(state);

        
    % Calculate the number of bins
    num_bins = floor(cols / bint);
    
    % Reshape the matrix to have 50 columns in each bin
    data_reshaped = reshape(state(:, 1:num_bins*bint), rows, bint, num_bins);
    
    % Take the mean along the second dimension (columns) of each bin
    binState = squeeze(mean(data_reshaped, 2));
    binned   = binState;%binState(:,1:tbin);

end

