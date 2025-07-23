%% Load Programmed RSA Data
output_dir = "../../hpc-outputs/programmed-networks";
runid = "28-Feb-2025_1000neurons_44a04e02-6fc5-4389-9109-584de368715b";

pth = fullfile(output_dir,runid);
statedir = "rnn-states";
svdir    = "rnn-analysis";
svname   = "rdm-vectors.mat";

if ~exist("brain","var"),load("../../data/brain_and_rishi_data.mat","brain"),end

% Load from file or compute if not yet run
if exist(fullfile(pth,svdir,svname),"file")
    load(fullfile(pth,svdir,svname))
else

    files   = dir(fullfile(pth,"*.mat"));
    nfiles  = numel(files); 
    progMat = cell(nfiles,1);
    
    mask = brain.all_mask(:,7:end);
    tpts = cellfun(@(s) find(s,1,"last"),num2cell(mask,2));
    
    for i = 1:nfiles
    
        a = load(fullfile(pth,files(i).name),"prnn_validation");
        prnn = a.prnn_validation;
        prnn.run_network;
       
        states = cellfun(@(x) x(:,5:end),...
            prnn.binned_states.modelStates,uniformoutput=false);
        
        ntrials  = numel(states);    
        stateCell = cell(ntrials,1);
        
        for j = 1:ntrials
            stateCell{j} = zscore(states{j}(:,1:tpts(j)),[],2);
        end
        
        progMat{i} = cat(2,stateCell{:});
        
    end

    rdmProg  = cellfun(@(s) pdist(s',"euclidean"),progMat,UniformOutput=false);

    % save(fullfile(pth,svdir,"state-matrices.mat"),"progMat")
    % save(fullfile(pth,svdir,svname),"rdmProg","-v7.3")
end


%%

load("../../data/brain_and_rishi_data.mat")
load("../../data/staticRDMs_brain_bbrnn.mat",'rdmBBs','rdmBrain');

rd   = rishi_data;
gtx = rd.sim_coordinates.x;
gty = rd.sim_coordinates.y;

mask     = brain.all_mask(:,7:end);
ntrials  = size(mask,1);
tpts     = cellfun(@(s) find(s,1,"last"),num2cell(mask,2))*50;

gtCell = cell(ntrials,1);

for i = 1:ntrials
    gtCell{i} = [
        gtx{i}(1:50:tpts(i))'
        gty{i}(1:50:tpts(i))'
        repmat(rd.xdot0(i),1,tpts(i)/50)
        repmat(rd.ydot0(i),1,tpts(i)/50)
        ];
end

gtCell = cat(2,gtCell{:});
rdmGT  = pdist(gtCell',"euclidean");


oraclecor = corr(rdmBrain', rdmGT', 'Type', 'Spearman');


%% Run Residualized Analysis 


oracle_gt_rdms  = rdmGT;
programmed_rdms = rdmProg;
neural_rdms     = rdmBrain;

corrtype = "Spearman";

rsaStruct_PANs  = static_residual_programmed(programmed_rdms, ...
    neural_rdms,oracle_gt_rdms,corrtype);

rsaStruct_oracle = static_residual_oracle({oracle_gt_rdms}, ...
    neural_rdms,programmed_rdms,corrtype);


%% Plot Results

plotBarChartComparisonRSA(rsaStruct_PANs, rsaStruct_oracle)


%% Plotting Function

function plotBarChartComparisonRSA(rsaStruct_PANs, rsaStruct_oracle)
% ---------------------------------------------------------------
% Seven-way violin plot (built-in violinplot)
% Order: Prog-Raw • Prog-NextTP • Prog-PaddleOnly • Prog-Both •
%        NextTP-Raw • NextTP-PaddleOnly • NextTP-Prog
% ---------------------------------------------------------------

% ---------- 1.  Collect column-vectors -------------------------
P = vertcat(rsaStruct_PANs);    % programmed model struct array

vals = { ...
    double([P.raw]'),                                      ... % Prog-Raw
    double([P.brain_minus_oracle_residual_oneRegressor]'), ... % Prog-Oracle
    double([rsaStruct_oracle.raw]'),                              ... % Oracle-Raw
    double([rsaStruct_oracle.brain_minus_PAN_residual_oneRegressor]')};%, ... % Oracle-1PAN
    % double([rsaStruct_oracle.brain_minus_PAN_residual_allRegressor])};  % Oracle-AllPAN    

% Common length (shorter vectors keep all rows, longer ones are clipped)
vals    = cat(2,vals{:});


% ---------- 2.  Long-form arrays for built-in violinplot -------
labels = {'PAN Raw','PAN-Lin.Map','Lin.Map Raw','Lin.Map-PAN'};

% ---------- 3.  Plot -------------------------------------------
figure;

xvals = [0 1 3 4 ];
b = bar(xvals,mean(vals)); b.BarWidth = .9;
ylabel('RSA (Spearman)');
title('RSA: Programmed vs Linear Map');
ylim([0 1]); 
xticklabels(labels)


% % ---------- 4.  Optional paired-star annotations ---------------
% pairs = [1 5; 2 7; 3 6];          % Raw, NextTP-resid, Paddle-resid comparisons
% addStars(gca, reshape(y, N, []), pairs);  %  existing helper

fx = @(y) randn(numel(y),1)*0.1;
hold on
ydata = {vals(:,1) vals(:,2) vals(1,3) vals(:,4)};
for i = [1 2 4]
    scatter(fx(ydata{i})+xvals(i),ydata{i});
end
hold off


end

%% Computing Functions
function staticRSA = static_residual_programmed(rdm_prnn, rdm_brain,regressors,corrtype,staticRSA)
%%  

    if ~exist("staticRSA","var"),staticRSA(1:2)=struct;end

    nmodels  = numel(rdm_prnn); 
    dBrain   = zscore(rdm_brain);

    dRegress = regressors';
    fcallAll = dRegress;

    dBrain_all = regress_out(dBrain, fcallAll());
    dBrain_one = zeros(numel(dBrain_all),nmodels);
    dModel     = zeros(size(dBrain_one));
    dModel_raw = cellfun(@zscore,rdm_prnn,UniformOutput=false);

    for i = 1:nmodels
        printloop(i,nmodels)
        dBrain_one(:,i) = regress_out(dBrain, dRegress);   
        dModel(:,i)     = regress_out(dModel_raw{i}, dRegress);
    end
    
    % rawTrig = ~strcmp(string(fieldnames(staticRSA)),"raw");
    parfor i = 1:nmodels

        printloop(i,nmodels)
        dModel   = dModel_raw{i}'; 
    
        [rsa_static_raw,pval_raw] = corr(dBrain', dModel, 'type',corrtype);
        [rsa_static_one,pval_one] = corr(dBrain_one(:,i), dModel, 'type',corrtype);
              
        staticRSA(i).("raw")      =  rsa_static_raw ; 
        staticRSA(i).("raw_pval") =  pval_raw ; 

        strid    = "brain_minus_oracle";
        staticRSA(i).(strid+"_residual_oneRegressor")      = rsa_static_one;  
        staticRSA(i).(strid+"_residual_oneRegressor_pval") = pval_one;  
        
        
    end

end

function staticRSA = static_residual_oracle(rdm_oracle, rdm_brain,regressors,corrtype,staticRSA)
%%  

    

    nmodels  = numel(regressors); 
    dBrain   = zscore(rdm_brain);

    if ~exist("staticRSA","var"),staticRSA(1:nmodels)=struct;end

    dRegress = zscore(vertcat(regressors{:})');
    fcall1   = @(x) dRegress(:,x);
    fcallAll = dRegress;


    dModel_raw = rdm_oracle{1}';
    dBrain_all = regress_out(dBrain, fcallAll());
    dBrain_one = zeros(numel(dBrain_all),nmodels);
    dModel_one = dBrain_one;
    dModel_all = regress_out(dModel_raw, fcallAll());
    

    for i = 1:nmodels
        printloop(i,nmodels)
        dBrain_one(:,i) = regress_out(dBrain, fcall1(i));   
        dModel_one(:,i) = regress_out(dModel_raw, fcall1(i));

        [rsa_static_raw,pval_raw]   = corr(rdm_brain', rdm_oracle{1}', 'type',corrtype);
        [rsa_static_one,pval_one]   = corr(dBrain_one(:,i), dModel_one(:,i), 'type',corrtype);
        [rsa_static_all,pval_all]   = corr(dBrain_all, dModel_all, 'type',corrtype);
              
        staticRSA(i).("raw")      =  rsa_static_raw ;
        staticRSA(i).("raw_pval") =  pval_raw ;
    
        strid    = "brain_minus_PAN";
        staticRSA(i).(strid+"_residual_oneRegressor")      =  rsa_static_one ;
        staticRSA(i).(strid+"_residual_oneRegressor_pval") =  pval_one ;                       
        staticRSA(i).(strid+"_residual_allRegressor")      =  rsa_static_all ;
        staticRSA(i).(strid+"_residual_allRegressor_pval") =  pval_all ; 
    end
    



end

function residualVec = regress_out(targetVec, regressors)

%% ========================================================================
% This function removes the linear contribution of one or more regressor
% vectors from a target vector by ordinary least squares. 
%
% Usage:
%   dBrain_res = regress_out(dBrain, dReg);
%   dBrain_res = regress_out(dBrain, [dReg1, dReg2, ...]);
%
% Inputs:
%   - targetVec: (N x 1) or (1 x N) vector of distances (e.g., from the brain)
%   - regressors: (N x k) matrix (or (1 x N) for single regressor) 
%                 where each column is a regressor distance vector.
%
% Output:
%   - residualVec: same size as targetVec, the residual after regressing out
%                  all columns in `regressors`.
%
% The formula is: residual = targetVec - X * (X \ targetVec),
% where X = [ones(N,1), regressors].
%% ========================================================================
    % Ensure column vectors
    targetVec = targetVec(:);  
    

    % If there are no regressors, just return targetVec
    if isempty(regressors), residualVec = targetVec;return;end

    % Check that sizes match
    if size(regressors,1) ~= length(targetVec)
        error('Target and regressors must have the same number of rows (trials).');
    end

    % Build design matrix X: first column is intercept (all ones),
    % then each column is one regressor
    X = [ones(size(regressors,1),1), regressors];

    % Solve for betas: b = X \ targetVec  (least squares)
    b = X \ targetVec;

    % Predicted = X*b
    predicted = X * b;

    % Residual = targetVec - predicted
    residualVec = targetVec - predicted;

end






