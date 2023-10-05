function train_and_run_rnn(rnnid , pausetime)
%% Load Dependencies

warning off

path = fileparts(matlab.desktop.editor.getActiveFilename());
cd(path)

rootpath = '../../';
addpath(genpath(rootpath))


%%

rng('shuffle')
pause(pausetime)

%% Add everything to path and start parallel pool


parpool(feature('NumCores'))

%% Load Monkey Data

load([rootpath 'data/perle_dataset.mat'],'perle')
load([rootpath 'data/mahler_dataset.mat'],'mahler')

mky1 = mahler.neural_responses_reliable;
mky2 =  perle.neural_responses_reliable;

brain = cat(1 , mky1 , mky2);

%% Get control files from Rishi

x = perle.meta.x0;
y = perle.meta.y0;
heading = perle.meta.heading0;
speed   = perle.meta.speed0;


%% Run for multiple networks with nonlin testing

nback = 1800;

n = 1000; runTime = 1800; dt = 0.01;

gameParams = {x , y , heading , speed*2};
runParams  = {n , runTime , dt };

nsplits  = 200;
normFunc = @(x , r) zscore(x - r, [] , 1);

clearvars netStats
netStats = struct();

% There is 20 time points to test and then a random position so the output
% vector for playing the game while decoding will be 20+1
tptvec = linspace(1 , 1000 , 20);
ntpts  = numel(tptvec);

%% Play Rishi Trials

try parpool(feature('NumCores')), catch, end

paramsv = runtrial_v2(gameParams , runParams);

% [RO , ROP, RPsv , WRPsv,~,xo, yw, ~,rs] = paramsv{:};

% yp = -(yw+xo);

%%
close all

figure('Position',[-3173 119 1197 987]), axis
for i = 1:79, plotTrial(paramsv, i , gcf),pause(.2),end


%%

[xcor , ~,wsx]  = decode_v2(nback, RPsv , WRPsv, rs, yp , normFunc);
xdata = 1:nback;

xcorsv = flipud(xcor);

[cor_fit , cor_gof] = testNonlinearity(xdata , xcorsv);
simsv               = testWithNeuralData(brain , RPsv , rs, nsplits);

net     = struct();
net.RO  = RO;
net.ROP = ROP;
net.wsx = wsx;

vars = struct();
vars.yw = yw;
vars.rs = rs;

[succStct, varout , statesv] = playGame(net, vars , tptvec , gameParams, normFunc);

simsv_game = zeros(nsplits , ntpts+1);
for i = 1:ntpts+1
    RPsv = cellfun(@(x) x(:,1:1800,i) , statesv, 'UniformOutput' , false)';
    simsv_game(:,i) = testWithNeuralData(brain , RPsv , rs, nsplits);   
end

netStats.xcorsv   = xcorsv;
netStats.nonlint  = cor_fit.c;
netStats.r2       = cor_gof.rsquare;
netStats.simavg   = mean(simsv);
netStats.simsv    = simsv;
netStats.simavg_game = mean(simsv_game);
netStats.simsv_game  = simsv_game;
netStats.success  = cat(2, succStct.succ{:});
netStats.varout   = varout;

fldr = 'hpcOutputs/';

save([fldr 'netStats_' num2str(rnnid) '.mat'] , "netStats" , '-v7.3')
save([fldr 'net_' num2str(rnnid) '.mat'] , 'net')
% save([fldr 'netGameState_' num2str(netid) '.mat'] , 'statesv', '-v7.3')

end

%% Support Functions

function [gamevars , logic] = runtrial_v2(gameParams, runParams)
%%

disp('Making RNN')
[x0 , y0 , heading , speed] = gameParams{:};
[n  , runTime , dt]          = runParams{:};

n = 3000;
m = 13; % Number of outputs for base RNN

% Define symbolic dynamics
xw =  0.05; % width for the arena
yw =  0.05; % height for the arena
yp = -0.05; % paddle height within the arena
pl =  0.02; % paddle length

syms t; assume(t,'real'); syms x(t) [m,1]; syms xc [m,1]; x = x(t); 
assume(x,'real'); assume(xc,'real');

% Phase bifrucation constants
fixedpoint = 0.1;
b = 3/13; 
a = -(1 + b) / (fixedpoint^2); 

% xf = 0.1; cx = 3/13; ax = -cx/(3*0.025^2);

% Velocities
xvel = x(1);
yvel = x(2);

% Positions
xpos = x(5)*xvel;
ypos = x(7)*yvel;

% Collision input, x(9:12)
collision_right   = 10*x(9)^2  - 2*fixedpoint ;
collision_left    = 10*x(10)^2 - 2*fixedpoint;
collision_top     = 10*x(11)^2 - 2*fixedpoint;
collision_paddle  = 50*x(12)^2 - 2*fixedpoint;

% Reccurent dynamics of the NAND gates to set up pitchfork phase diagram
% and over all dynamic behavior
nand_reccurent = a*x(5:8).^3 + b*x(5:8) - fixedpoint;

% NAND-gate logic with input from collision and from the paired NAND in sr-latch 
% this value will shift the reccurent phase diagram of the system
nand_logic = [
    ( collision_right  * ( x(6) - fixedpoint ) )
    ( collision_left   * ( x(5) - fixedpoint ) )
    ( collision_top    * ( x(8) - fixedpoint ) )
    ( collision_paddle * ( x(7) - fixedpoint ) )
    ]./(2*fixedpoint);

% Connect the sr-latches
srx = [
    20*(nand_reccurent(1) + nand_logic(1))
    20*(nand_reccurent(2) + nand_logic(2))
    ];

sry = [
    20*(nand_reccurent(3) + nand_logic(3))
    20*(nand_reccurent(4) + nand_logic(4))
    ];

% Board/Arena centered at (0,0)
% outputs a fixed point of 0 if linear term on reccurent x(#) = 0, ie -ax^3 + bx | b = 0
% else, the fixed point bifrucates and splits to both a +/- stable fixed
% point pair, thus making the detector trigger the sr-latch, thus moving the
% ball back towards the center and having the collision detector go back to 0
collision_detector_right  = 1000*( ( x(3) - xw )*x(09) - x(09)^3); % if  x(3) <  x-width , linear term = 0
collision_detector_left   = 1000*( (-x(3) - xw )*x(10) - x(10)^3); % if -x(3) < -x-width , linear term = 0
collision_detector_top    = 1000*( ( x(4) - yw )*x(11) - x(11)^3); % if  x(4) <  y-width , linear term = 0  

% Paddle collision is slightly different in that the paddle position is also 
% a variable, x(13);
collision_logic_paddle    = (pl - ( (x(3) - x(13) )^2 + ( x(4) - yp )^2) );
collision_detector_paddle = 2000*( collision_logic_paddle*x(12) - x(12)^3);

% Define complete state-update:
% each row will be added to the variable oneach subsequent time-step
dx = [0 % d(dx) velocity (no 𝛥 , constant velosity)
      0 % d(dy) velocity (no 𝛥 , constant velosity)
      xpos % dx position (gets signed xvelocity and magnitude added to update position)
      ypos % dy position (gets signed yvelocity and magnitude added to update position)
      srx  % srlatch for left/right wall collisions (flips sign of xvelocity)
      sry  % srlatch for top/paddle wall collisions (flips sign of yvelocity)
      collision_detector_right  % x( 9)
      collision_detector_left   % x(10)
      collision_detector_top    % x(11)
      collision_detector_paddle % x(12)
      0 % paddle (outside input, not folded into recurrent model dynamics)
      ];

% Decompile base RNN into basis set
basernn = getexpansion(n , m, dt);

% Uses symbolic programmed logic to compile W for base RNN
W = getW(dx,x,basernn);

% Run Through Trials
ntrials = numel(x0);

A   = basernn.A;
B   = basernn.B;
rs  = basernn.rs;
xs  = basernn.xs;
gam = basernn.gam;

RPsv  = cell(ntrials,1);
WRsv  = cell(ntrials,1);
prsv  = cell(ntrials,1);

basernn_par = repmat(basernn , ntrials , 1);

% for i = randi(79)
parfor i = 1:ntrials
    
    yfunc = @(x) (x - 16)/266.6666;

    xstart =  yfunc(y0(i));
    ystart = -yfunc(x0(i));
    
    yvel0 = ( -cos(-heading(i))/sqrt(266.6666) ) * speed(i);
    xvel0 = ( -sin(-heading(i))/sqrt(266.6666) ) * speed(i);
    
    % xstart = -.05;
    % ystart = -.02;
    % 
    % yvel0 = .1;
    % xvel0 = .1;

    padl = [0 0]-0.05; % [startpos endpos]    

    % % Train Ball Rnn
    % basenn = ReservoirTanhB_noDisp(A,B,rs,xs,dt,gam); d = RO.d;
    
    % Initialize burnin of base RNNs to get the appropriate starting state
    % for these initial conditions
    init_conds = [
        abs(xvel0) ; abs(yvel0)
        xstart ; ystart
        sign(xvel0)*0.1;-sign(xvel0)*0.1
        sign(yvel0)*0.1;-sign(yvel0)*0.1
        0;0;0;0;padl(1)
        ];

    % RT = RO.train(repmat(init_conds,[1,4000,4]));
    
    states_train = basernn_par(i).train(repmat(init_conds,[1,4000,4]));
    
    reccurent_idx = 1:12;
    outside_idx   = 13;

    BW = B(:,reccurent_idx)*W(reccurent_idx,:);
    Bx = B(:,outside_idx);

    % Compile Programmed RNN
    prnn = ReservoirTanhB_noDisp(A+BW,Bx,rs,0,dt,gam);
    prnn.d = basernn.d; prnn.r = states_train(:,end);
    
    prsv{i} = states_train(:,end);

    % Drive/Evolve with paddle inputs and play game
    t1 = runTime; % hardcoded multiplier (30 frames of our time is like 1 frame of theirs)
    xt = linspace(padl(1),padl(2),t1);
    
    network_game = prnn.trainSVD(repmat(xt,[1,1,4]),rank(W)); 

    % Render Output
    WR = W*network_game;
    
    RPsv{i}  = network_game; 
    WR_sv{i} = WR; 

end

% Train Ball Rnn
RO = ReservoirTanhB_noDisp(A,B,rs,xs,dt,gam); d = RO.d;

% % % Initialize burnin of base RNNs to get the appropriate starting state
% % % for these initial conditions
% % init_conds = [
% %     abs(xvel0) ; abs(yvel0)
% %     xstart ; ystart
% %     sign(xvel0)*0.1;-sign(xvel0)*0.1
% %     sign(yvel0)*0.1;-sign(yvel0)*0.1
% %     0;0;0;0;padl(1)
% %     ];

% % RT = RO.train(repmat(init_conds,[1,4000,4]));

reccurent_idx = 1:12;
outside_idx   = 13;

BW = B(:,reccurent_idx)*W(reccurent_idx,:);
Bx = B(:,outside_idx);

% Compile Programmed RNN
ROP = ReservoirTanhB_noDisp(A+BW,Bx,rs,0,dt,gam);
% % ROP.d = d; ROP.r = RT(:,end);
xo    = 0.01;


% scatter(WRP(3,:),WRP(4,:))
logic.xo = xo;
logic.yw = yw;
logic.xw = xw;
logic.yp = yp;
logic.pl = pl;

gamevars = { basernn , ROP, RPsv , WRsv , ntrials , prsv };


end

function [basernn] = getexpansion(n_neurons , n_outputs, dt)
%%

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

%% Archive

function params = runtrial(gameParams, runParams)

disp('Making RNN')
[x0 , y0 , heading , speed] = gameParams{:};
[n  , runTime , dt]          = runParams{:};

% Number of start positions
ntrials = numel(x0);

% Inputs
m = 13;

% Time
gam = 100;

% Reservoir
% Define reservoir
A  = sparse(zeros(n));
B  = (rand(n,m)-.5)*.1;
rs = (rand(n,1)-.5);
xs = zeros(m,1);
RO = ReservoirTanhB_noDisp(A,B,rs,xs,dt,gam); d = RO.d;

% Fixed points for new shift
RO.r = rs; rsT = rs;

% Decompile (Get the basis set)
dv       = A*rsT + B*xs + d;
[Pd1,C1] = decomp_poly1_ns(A,B,rsT,dv,4);

% Compute shift matrix
[Pdx,Pdy] = find((sum(Pd1,2)+sum(Pd1,2)') <= max(sum(Pd1,2)));
PdS       = zeros(size(Pd1,1));

for i = 1:length(Pdx)
    PdS(Pdx(i),Pdy(i)) = find(sum(Pd1==(Pd1(Pdx(i),:)+Pd1(Pdy(i),:)),2)==m);
end

% Finish generating basis
Aa = zeros(size(C1));
Aa(:,(1:m)+1)  = Aa(:,(1:m)+1)+B;  Aa(:,1) = Aa(:,1) + d;
GdNPL = gen_basis(Aa,PdS);

% Define symbolic dynamics
xw = 0.05; yw =  0.05;
xp = 0.05; yp = -0.05;
syms t; assume(t,'real'); syms x(t) [m,1]; syms xc [m,1]; x = x(t); 
assume(x,'real'); assume(xc,'real');

xf = 0.1; cx = 3/13; ax = -cx/(3*0.025^2);
dx = [x(2)*x(12);... 1
      20*(-.1 + (10*x(7)^2-xf -xf)*( x(3) -xf)/(2*xf)  + cx*x(2)  + ax*x(2)^3);... 2
      20*(-.1 + (10*x(8)^2-xf -xf)*( x(2) -xf)/(2*xf)  + cx*x(3)  + ax*x(3)^3);... 3
      x(5)*x(13);... 4
      20*(-.1 + (10*x(9)^2-xf -xf)*( x(6) -xf)/(2*xf)  + cx*x(5)  + ax*x(5)^3);... 5
      20*(-.1 + (50*x(10)^2-xf-xf)*( x(5) -xf)/(2*xf)  + cx*x(6)  + ax*x(6)^3);... 6
      1000*(( x(1) -xw )*x(7) - x(7)^3);... 7
      1000*((-xw   -x(1))*x(8) - x(8)^3);... 8
      1000*(( x(4)-yw)*x(9) - x(9)^3);... 9
%       1000*((-yw-x(4))*x(10) - x(10)^3);... 10
      2000*((.002-((x(1)-x(11))^2+(x(4)-yp)^2))*x(10) - x(10)^3);
      0;... 11
      0;... 12
      0]; % 13

% Convert symbolic output to source code by extracting coefficients
pr        = primes(2000)'; pr = pr(1:m);
[~,DXC] = sym2deriv(dx,x,pr,Pd1,PdS);
o         = zeros(m,size(C1,2)); o(:,(1:m)+1) = eye(m);
oS        = DXC;
OdNPL     = o+oS/gam;

% Compile
W = lsqminnorm(GdNPL', OdNPL')';

s   = whos;
idx = strcmp({s.class} , 'sym');
clear(s(idx).name)
clearvars s

%%% Run Through Trials
RPsv  = cell(ntrials,1);
WRPsv = cell(ntrials,1);
% rsSv  = cell(ntrials,1);

% parfor i = 1:ntrials
parfor i = 1:ntrials
    

    yfunc = @(x) (x - 16)/266.6666;

    xstart = yfunc(y0(i));
    ystart = -yfunc(x0(i));
    
    yvel0 = ( -cos(-heading(i))/sqrt(266.6666) ) * speed(i);
    xvel0 = -sin(-heading(i))/sqrt(266.6666) * speed(i);
    
    padl = [0 0]+0.09; % [startpos endpos]    

    % Train Ball Rnn
    RO = ReservoirTanhB_noDisp(A,B,rs,xs,dt,gam); d = RO.d;
    
    RT = RO.train(repmat([xstart;sign(xvel0)*0.1;-sign(xvel0)*0.1;...
                          ystart;sign(yvel0)*0.1;-sign(yvel0)*0.1;...
                          0;0;0;0;padl(1);...
                          abs(xvel0);abs(yvel0)],[1,4000,4]));
    
    ROP = ReservoirTanhB_noDisp(A+B(:,[1:m-3 m-1:end])*W([1:m-3 m-1:end],:),B(:,m-2),rs,0,dt,gam);
    ROP.d = d; ROP.r = RT(:,end);
    
%     rsSv{i} = RT(:,end);

    % Drive with Paddle and play game
    t1 = runTime; % hardcoded multiplier (30 frames of our time is like 1 frame of theirs)
    xt = linspace(padl(1),padl(2),t1);
    
    RP = ROP.trainSVD(repmat(xt,[1,1,4]),rank(W)); 

    % Output
    WRP = W*RP;
    

    RPsv{i}  = RP; 
    WRPsv{i} = WRP; 

end

RO    = ReservoirTanhB_noDisp(A,B,rs,xs,dt,gam);
ROP   = ReservoirTanhB_noDisp(A+B(:,[1:m-3 m-1:end])*W([1:m-3 m-1:end],:),B(:,m-2),rs,0,dt,gam);
ROP.W = W;
xo    = 0.01;

params = {RO , ROP, RPsv , WRPsv , ntrials , xo, yw, xw , rsT};


end

function [xcor,ycor,wsave,xerr] = decode_v2(nback , RPsv , WRPsv , rs, yp,normFunc)
disp('Decoding Final Position')
ntrials   = numel(RPsv);
[n,ntpts] = size(RPsv{1});

[fxpos , fypos] = deal(nan(ntrials , 1));
states  = zeros(n , ntrials , nback);

useBoardend = true;

for i = 1:ntrials

    if iscell(rs) , rstar = rs{i}; else, rstar = rs; end
    a = normFunc(RPsv{i},rstar); % states (n , ntpts)
    b = WRPsv{i}; % 

    if useBoardend 
        fnidx  = find(b(4,:) < yp,1);
        if isempty(fnidx),fnidx  = size(b,2);end
    else
        fnidx = ntpts;
    end

    if fnidx < nback, continue,end

    grabidx = (fnidx-nback)+(1:nback);
           
    fxpos(i)  = b(1 ,fnidx);
    fypos(i)  = b(4 ,fnidx);
    states(:,i,:) = fliplr(a(: , grabidx));  
  
% %     grabidx = 1:nback;
% % 
% %     fxpos(i)  = b(1 ,end);
% %     fypos(i)  = b(4 ,end);
% %     states(:,i,:) = fliplr(a(: , grabidx));  

end

idx = isnan(fxpos);

fxpos(idx) = [];
fypos(idx) = [];
states(:,idx,:) = [];

ntrials = sum(~idx);

trainidx = randi(ntrials , floor(ntrials/2) , 1);
bin      = true(ntrials,1); bin(trainidx) = false;
vec      = 1:ntrials;
valididx = vec(bin);

xfpostr =  fxpos(trainidx);
yfpostr =  fypos(trainidx);
statetr = states(:,trainidx,:);

xfposvd =  fxpos(valididx);
yfposvd =  fypos(valididx);
statevd = states(:,valididx,:);

[xerr, yerr, xcor,ycor] = deal(zeros(nback , 1));
[xsave,ysave] = deal(zeros(numel(valididx) , nback));
[wsvx, wsvy] = deal(zeros(n,nback));
parfor i = 1:nback

    state = statetr(:,:,i)';

    stinv = pinv(state);

    wsx = stinv*xfpostr;
    wsy = stinv*yfpostr;

    wsvx(:,i) = wsx;
% %     wsvy(:,i) = wsy;

    state = statevd(:,:,i)';

    vdx = state*wsx;
    vdy = state*wsy;

    xsave(:,i) = vdx;
% %     ysave(:,i) = vdy;
% % 
    xerr(i) = norm(vdx - xfposvd)/size(state,1);
% %     yerr(i) = norm(vdy - yfposvd)/size(state,1);

    xcor(i) = corr(vdx, xfposvd,'Type','Pearson');
    ycor(i) = corr(vdy, yfposvd,'Type','Pearson');


end

x.xpred = xsave;
x.xtrue = xfposvd;
x.xerr  = xerr;
wsave   = wsvx;

end

function [f1 , gof] = testNonlinearity(xdata , ydata)
disp('Getting Nonlinearity Fit')

% Set Equation
sigmoidEqn = fittype('(a/(1+exp(-b*(x-c)))) + d');

opts  = fitoptions(sigmoidEqn);
% opts_pos = fitoptions(sigmoidEqn);

opts.StartPoint = [5  .2 180   0 ];
opts.Lower      = [0  .1  0    0 ];
opts.Upper      = [20 .5 500  15 ];
% opts.Exclude    = newx > 500;

% Fit
[f1, gof] = fit(xdata',ydata,sigmoidEqn,opts);

% % % Calc New Curve
% % fity = feval(f1, newx);


end

function simsv = testWithNeuralData(brain , rnn , rs,  nruns)
disp('Comparing with Neural Data')

simsv = zeros(nruns , 1);
count = 1;

mdlStates = permute(cat(3, rnn{:}) , [1 3 2]);

neuralidx = 1:26;
bint      = 36;


mdl   = modelBin(mdlStates , bint);

m1 = mdl(:,:,neuralidx) - rs;
m2 = brain(:,:,neuralidx);

m1 = reshape(m1 , [], prod(size(m1 , [2 3])))';
m2 = reshape(m2 , [], prod(size(m2 , [2 3])))';

m = squareform(pdist(m1));
b = squareform(pdist(m2));

nstates = size(m,1);

while count <= nruns
   
    bin  = false(nstates,1);
    bin(randperm(nstates , nstates/2)) = true;
    
    mm1 = m(bin,:);
    mm2 = m(~bin,:);
    
    bin  = false(nstates,1);
    bin(randperm(nstates , nstates/2)) = true;
    
    bb1 = b(bin,:);
    bb2 = b(~bin,:);
    
    pmb = corrcoef(m , b);
    pmm = corrcoef(mm1 , mm2);
    pbb = corrcoef(bb1 , bb2);
    
    sim = pmb(2) / sqrt(pmm(2)*pbb(2));
    
    if isreal(sim), simsv(count) = sim; 
        count = count+1;end

end


end

function binnedMdl = modelBin(RP , bint)

    [n , ncond , ntpts] = size(RP);

    idx = reshape(1:ntpts , bint , [])';
    nbins = size(idx,1);

    binnedMdl = zeros(n , ncond , nbins);
    for j = 1:nbins
        binnedMdl(:,:,j) = mean(RP(: , :, idx(j,:)),3);  
    end

   


end

function [succStct, varout , statesv] = playGame(net, vars , tptvec , gameParams , normFunc)
%%
disp('Playing Game')


wsx = net.wsx;


rs = vars.rs;

[x0,y0,heading,speed] = gameParams{:};

ntrials = numel(x0);
yfunc   = @(x) (x - 16)/266.6666;

succStct = struct();

ntpts  = numel(tptvec);
t      = 1:size(wsx,2);
stasv  = cell(ntrials,1);
succsv = cell(ntrials,1);
varout(1:ntrials) = struct();

try parpool('numCores'), catch, end

parfor i = 1:ntrials

    RO  = net.RO;
    ROP = net.ROP;

    W   = ROP.W;

    disp(['   trial ' num2str(i) '/' num2str(ntrials)])
    xstart = yfunc(y0(i));
    ystart = -yfunc(x0(i));

    nneur  = size(ROP.r,1);
    tmax   = 180; % Maximum time to allow for simulation

    xo      = 0.01;
    bordend = -(vars.yw+xo);
    
    dt    = ROP.delT;
    t1    = .1/dt;
    px    = 0; % Initial paddle position

    yvel0 = ( -cos(-heading(i))/sqrt(266.6666) ) * speed(i);
    xvel0 = -sin(-heading(i))/sqrt(266.6666) * speed(i);

    padl = [0 0]+vars.yw+.03; % [startpos endpos]    
    
    mag = 0.1;

    % Train Ball Rnn
    d    = RO.d;     
    RO.r = rs;
    RT   = RO.train(repmat([xstart;sign(xvel0)*mag;-sign(xvel0)*mag;...
                      ystart;sign(yvel0)*mag;-sign(yvel0)*mag;...
                      0;0;0;0;padl(1);...
                      abs(xvel0);abs(yvel0)],[1,1000,4]));
    
    succ  = zeros(ntpts+1,1);
    state = zeros(nneur , t1*tmax , ntpts+1);
    for j = 1:ntpts

        disp(['        tpt ' num2str(j) '/' num2str(ntpts)])
        [~,tpt] = min(abs(t - tptvec(j))) ;

        ROP.d   = d; ROP.r = RT(:,end);
                
        
        % Drive with Paddle and play game 
        
        out   = zeros(10);
        count = 1; 
        xt    = linspace(px,px,t1);
        
        [xd,yd,pd] = deal(nan(tmax,1));
                
        while out(4,end) > bordend && count <= tmax
        
            pd(count) = px;
        
            RP = ROP.trainSVD(repmat(xt ,[1,1,4]),rank(W));
            state(: , (1:t1)+((count-1)*t1),j) = RP;
            
            out   = W*RP;

            normRP = normFunc(RP(:,end) , rs);
            xpred  = normRP'*wsx(:,tpt);
        
            xd(count) = out(1,end);
            yd(count) = out(4,end);
        
            if abs(xpred) > 0.03, xpred = px; end
            px = xpred;
            xt = linspace(px,px,t1);    
            
            count = count+1;
        
        end
                
        if (count-1) == tmax, succ(j) = 1; end

    end

    disp('        random ')

    ROP.d   = d; ROP.r = RT(:,end);
                
    % Drive with Paddle and play game     
    out   = zeros(10);
    count = 1; 
    xt    = linspace(px,px,t1);    
            
    while out(4,end) > bordend && count <= tmax
       
        RP = ROP.trainSVD(repmat(xt ,[1,1,4]),rank(W));
        state(: , (1:t1)+((count-1)*t1), j+1 ) = RP;
        
        px = 0.06*(2*rand()-1);
        xt = linspace(px,px,t1);    
        
        count = count+1;
    
    end

    if (count-1) == tmax, succ(j+1) = 1; end

    succsv{i}  = succ;
    statesv{i} = state;
    
    varout(i).xd = xd; % ball xdata
    varout(i).yd = yd; % ball ydata
    varout(i).pd = pd; % ball paddle data
    varout(i).xo = xo; % offset data for walls
    varout(i).tptvec = tptvec;

end

    succStct.succ    = succsv;    


end

function sandbox()
figure, axis
for i = 1:6, plotTrial(paramsv, i , gcf),pause(1),end

%%

figure, axis
for i = 1:79, plotTrial(paramsv, i , gcf),pause(1),end

end
function plotTrial(params,idx , f)

[~ , ~, ~ , WRPsv , ntrials , xo, yw, xw ,yp, ~] = params{:};

WRPsv = WRPsv{idx};

% figure('Position',[-2470 85 851 823]); hold on
if ~exist('f','var')
f = figure('Position',[570 328 892 748]);gca
end
ax = f.Children(1);
set(ax,'visible',0);

% Plot Wall

plot(ax, -1*[-1 1 1 -1]*(yw+xo), -1*[-1 -1 1 1]*(xw+xo),'k-', 'linewidth',4,...
     'clipping',0);

hold(ax,'on')


bordend = -(yw+xo);

ballsz = 30;

clrs = parula(ntrials);

useBoardend = false;
for i = idx

    b = WRPsv;

%     fnidx  = find(b(4,:) < bordend,1);
%     fnidx  = size(b,2);
    if useBoardend 
        fnidx  = find(b(4,:) < bordend,1);
    else
        fnidx = size(b,2);
    end

    if isempty(fnidx),fnidx  = size(b,2);end
           
    y = -b(3,1:fnidx);
    x = -b(4,1:fnidx);
    nx = numel(x);

    grey = [linspace(255, 100, nx)', linspace(255, 100, nx)', linspace(255, 10, nx)'];
    clr  = (clrs(i,:).*grey)/255;

    % Ball   
    scatter(ax, x(1), y(1), ballsz, clr(1,:), 'filled',...
        'MarkerFaceAlpha',0.3,'MarkerEdgeColor','r','LineWidth',2); 
    scatter(ax, x, y, ballsz, clr, 'filled','MarkerFaceAlpha',.3); 

    py = linspace(-b(13,end)-.002 , -b(13,end)+.002,100);
    px = repmat(-yp,size(py,2),1);
    
    scatter(ax,px ,py , ballsz, 'k', "square",'filled'); 

end

set(ax, 'XLim' , [-1 1]*yw*2 , 'YLim' , [-1 1]*xw*2 , 'Visible' , false)

hold(ax,'off')


end




