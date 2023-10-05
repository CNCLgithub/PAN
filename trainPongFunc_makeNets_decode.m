function trainPongFunc_makeNets_decode(nnets , nback , ntrials)
%{

This function creates nnets rnns that are programmed to 
simulate the game of pong. It then uses linear regression to
make a map between the neural-state at time-point k, relative 
to its final position (the end of the game board slash where
the paddle is).

It will make this prediction from ntrials/2, and validate this
prediction using the correlation between the actual final position
and the predicted position of the heldout trials.

It then saves this information about neural-states, validation,
trials, and decoding/predition mapping in a structure which
is saved as ./pongNets.mat.

This structure is loaded by:
    runDSQ_pong.sh >> pong.txt >> trainPongFunc_playGame.m
to have networks actually play the game using these 
decoding/prediction maps to move the paddle.

%}
%%

addpath(genpath('.'))
parpool(feature('NumCores'))

%%

if 1
    nnets   =    1;
    nback   = 5000;
    ntrials =  1;
end

xcor  = zeros(nback , nnets);
wcell = cell(nnets,1);

gameParams.headingRange = [-1 1]*1.2;
gameParams.xStartRange  = [4 18];
gameParams.yStartRange  = [1.6 30];
gameParams.dt           = 0.01;
gameParams.gamma        = 100;
gameParams.speedRange   = 1:3;

clearvars nets
nets(1:nnets) = struct();
[stateSV , posSV] = deal(cell(nnets,1));
for i = 1:nnets
    
    disp([num2str(i) '/' num2str(nnets)])
    
    [RPsv , WRPsv, vars] = programRNN(ntrials , gameParams);
    
    nets(i).RO  = vars.RO;
    nets(i).ROP = vars.ROP;
    nets(i).W   = vars.W;
    nets(i).id  = i;

    [xcor(:,i) , ~,wsave,~] = decode(nback , RPsv , WRPsv,vars);

    wcell{i} = wsave;
end

vars.wsx  = wcell;
vars.xcor = xcor;

pongNets.nets = nets;
pongNets.vars = vars;

pongStatesTrain.neural = RPsv;
pongStatesTrain.simul  = WRPsv;

%% Save 

save('pongNets.mat',"pongNets",'-v7.3')
save('pongStatesTrain.mat',"pongStatesTrain",'-v7.3')

end
%% Functions 

function [RPsv , WRPsv , vars] = programRNN(ntrials , gameParams)
%% Run network and train for ntrial dirfferent starting positions
% Global parameters

% Number of start positions
% ntrials = 100;

% Neurons
m = 13;
n = 1000;

% Time
dt  = 0.01;
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
dim = 0.03;

xw = dim; yw =  dim;
xp = dim; yp = -0.04;
syms t; assume(t,'real'); syms x(t) [m,1]; syms xc [m,1]; x = x(t); 
assume(x,'real'); assume(xc,'real');

% Logic
xf = 0.1; cx = 3/13; ax = -cx/(3*0.025^2);
dx = [x(2)*x(12);...
      20*(-.1 + (10*x(7)^2-xf -xf)*( x(3) -xf)/(2*xf)  + cx*x(2)  + ax*x(2)^3);...
      20*(-.1 + (10*x(8)^2-xf -xf)*( x(2) -xf)/(2*xf)  + cx*x(3)  + ax*x(3)^3);...
      x(5)*x(13);...
      20*(-.1 + (10*x(9)^2-xf -xf)*( x(6) -xf)/(2*xf)  + cx*x(5)  + ax*x(5)^3);...
      20*(-.1 + (50*x(10)^2-xf-xf)*( x(5) -xf)/(2*xf)  + cx*x(6)  + ax*x(6)^3);...
      1000*(( x(1)-xw)*x(7) - x(7)^3);...
      1000*((-xw-x(1))*x(8) - x(8)^3);...
      1000*(( x(4)-yw)*x(9) - x(9)^3);...
%       1000*((-yw-x(4))*x(10) - x(10)^3);...
      2000*((.002-((x(1)-x(11))^2+(x(4)-yp)^2))*x(10) - x(10)^3);...
      0
      0
      0];

% Convert symbolic output to source code by extracting coefficients
pr        = primes(2000)'; pr = pr(1:m);
[~,DXC]   = sym2deriv(dx,x,pr,Pd1,PdS);
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

runt = 30; % time in seconds to run forwards
try parpool(), catch, end

    parfor i = 1:ntrials
    
        xstart = 2*rand(1)-1; xstart = xstart*(xw+.01);
        ystart = 0.03 + 0.001*randn(1);     
    
        mag   = .1;
        thet  =  (rand(1)*.8 + (1-.8)/2 )*pi;
        xvel0 =  cos(thet)*mag;
        yvel0 = -sin(thet)*mag;
       
        padl = [0 0]+xw+.03; % [startpos endpos]    
        
        % Train Ball Rnn
        RO = ReservoirTanhB_noDisp(A,B,rs,xs,dt,gam); d = RO.d;     
        
        RT = RO.train(repmat([xstart;sign(xvel0)*mag;-sign(xvel0)*mag;...
                          ystart;sign(yvel0)*mag;-sign(yvel0)*mag;...
                          0;0;0;0;padl(1);...
                          abs(xvel0);abs(yvel0)],[1,1000,4]));
        
        ROP = ReservoirTanhB_noDisp(A+B(:,[1:m-3 m-1:end])*W([1:m-3 m-1:end],:),B(:,m-2),rs,0,dt,gam);
%         ROP = ReservoirTanhB_noDisp(A+B(:,[1:m-3])*W([1:m-3],:),B(:,m-2:end),rs,0,dt,gam);
        ROP.d = d; ROP.r = RT(:,end);
        
        % Drive with Paddle and play game
        t1 = runt/dt; 
        xt = linspace(padl(1),padl(2),t1);

        RP = ROP.trainSVD(repmat(xt ,[1,1,4]),rank(W)); 
        
        % Output
        WRP = W*RP;
        
        RPsv{i}  = RP; 
        WRPsv{i} = WRP; 
    
    end

    vars.xw = xw;
    vars.yw = yw;
    vars.xp = xp;
    vars.yp = yp;

    vars.xf = xf;
    vars.cx = cx;
    vars.ax = ax;

    vars.W   = W;
    vars.RO  = ReservoirTanhB_noDisp(A,B,rs,xs,dt,gam);
    vars.ROP = ReservoirTanhB_noDisp(A+B(:,[1:m-3 m-1:end])*W([1:m-3 m-1:end],:),B(:,m-2),rs,0,dt,gam);

end

function [xcor,ycor,wsave,x] = decode(nback , RPsv , WRPsv,vars)

ntrials   = numel(RPsv);
[n,ntpts] = size(RPsv{1});

[fxpos , fypos] = deal(zeros(ntrials , 1));
states  = zeros(n , ntrials , nback);

for i = 1:ntrials

    a =  RPsv{i};
    b = WRPsv{i};

    fnidx  = find(b(4,:) < vars.yp,1);
    if isempty(fnidx),fnidx  = size(b,2);end

    grabidx = (fnidx-nback+1):fnidx;
           
    fxpos(i)  = b(1 ,fnidx);
    fypos(i)  = b(4 ,fnidx);
    states(:,i,:) = fliplr(a(: , grabidx));   

end

tridx    = floor(ntrials/2);
trainidx = 1:tridx;
valididx = tridx+1:ntrials;

xfpostr =  fxpos(trainidx);
yfpostr =  fypos(trainidx);
statetr = states(:,trainidx,:);

xfposvd =  fxpos(valididx);
yfposvd =  fypos(valididx);
statevd = states(:,valididx,:);

[xerr, yerr, xcor,ycor] = deal(zeros(nback , 1));
[xsave,ysave] = deal(zeros(tridx , nback));
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

