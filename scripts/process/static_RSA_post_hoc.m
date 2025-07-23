%% Load Programmed RSA Data

% load PAN model run from hpc-ouput/
output_dir = "../../hpc-outputs/programmed-networks";
runid = "28-Feb-2025_1000neurons_44a04e02-6fc5-4389-9109-584de368715b";

pth = fullfile(output_dir,runid);
statedir = "rnn-states";
svdir    = "rnn-analysis";
svname   = "rdm-vectors.mat";

if ~exist("brain","var"),load("../../data/brain_and_rishi_data.mat","brain"),end

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

    save(fullfile(pth,svdir,"state-matrices.mat"),"progMat")
    save(fullfile(pth,svdir,svname),"rdmProg","-v7.3")
end


%% Load Brain and Black Box RSA Data

load("../../data/bbrnns.mat","bbrnns")

if exist("../../data/staticRDMs_brain_bbrnn.mat","file")
    % These should exist already so this should work but if for some reason
    % it doesn't it will generate the RDM vectors needed de novo
    load("../../data/staticRDMs_brain_bbrnn.mat")
else
    braindata = zscore(brain.data(:,:,7:end-1),[], 3);
    brainData = num2cell(braindata,[1 3])';
    brainData = cellfun(@squeeze,brainData,UniformOutput=false);
    
    bbData = cellfun(@(s) num2cell(zscore(double(s),[],3),[1 3]),{bbrnns.data_neur_nxcxt}',uniformoutput=false);
    bbData = cat(1,bbData{:});
    bbData = cellfun(@squeeze,bbData,UniformOutput=false);
    mask = brain.all_mask(:,7:end);
    
    ntrials = size(mask,1);
    tpts = cellfun(@(s) find(s,1,"last"),num2cell(mask,2));
    
    stateCell = cell(ntrials,1);
    brainCell = stateCell;
    bbCell    = stateCell;
    
    for i = 1:ntrials
        brainCell{i} = brainData{i}(:,1:tpts(i));
        bbCell{i}    = cellfun(@(s) s(:,1:tpts(i)),bbData(:,i),UniformOutput=false );
    end
    
    bbCell = cat(2,bbCell{:});
    
    bbMats = cellfun(@(s) horzcat(s{:}) , num2cell(bbCell,2),UniformOutput=false);
    brainMat = cat(2,brainCell{:});
    
    rdmBBs   = cellfun(@(s) pdist(s',"euclidean"),bbMats,UniformOutput=false);
    rdmBrain = pdist(brainMat',"euclidean");

    save("../../data/staticRDMs_brain_bbrnn.mat","rdmBBs","rdmBrain",'-v7.3')

end


%% Run Raw (no residualization) Correlations

bbrnncor = zeros(numel(rdmBBs),1);
prnncor  = zeros(numel(rdmProg),1);

% Create a parallel pool if it doesn't already exist
pool = gcp();

% --- First Loop with parfeval ---
nBB = numel(rdmBBs);
futuresBB = parallel.FevalFuture.empty(nBB,0);

for i = 1:nBB
    futuresBB(i) = parfeval(pool, @computeCorr, 1, rdmBrain, rdmBBs{i}, i, nBB);
end

% Collect results as they finish
for i = 1:nBB
    [idx, val] = fetchNext(futuresBB);
    bbrnncor(idx) = val;
end

% --- Second Loop with parfeval ---
nP = numel(rdmProg);
futuresP = parallel.FevalFuture.empty(nP,0);

for i = 1:nP
    futuresP(i) = parfeval(pool, @computeCorr, 1, rdmBrain, rdmProg{i}, i, nP);
end

for i = 1:nP
    printloop(i,nP)
    [idx, val] = fetchNext(futuresP);
    prnncor(idx) = val;
end

% --- Local Function ---
function out = computeCorr(rdmBrain, rdmX, idx, total)
    printloop(idx, total);
    out = corr(rdmBrain', rdmX', 'Type', 'Spearman');
end


%% Run Programmed Models' RSA Residuals

losskey = contains(string({bbrnns.loss_type}),"-mov");
neurkey = cellfun(@(s) size(s,1),{bbrnns.data_neur_nxcxt}) == 40;
inpkey  = contains(string({bbrnns.input_representation}),"gabor");

nextp   = losskey & neurkey & inpkey;
pdlonly = ~losskey & neurkey & inpkey;

nextTimePoint_rdms = rdmBBs(nextp);
paddleOnly_rdms    = rdmBBs(pdlonly);
programmed_rdms    = rdmProg;
neural_rdms        = rdmBrain;

corrtype = "Spearman";

rsaStruct = static_residual_programmed(programmed_rdms, ...
    neural_rdms,"nextTimePoint",nextTimePoint_rdms,corrtype);

rsaStruct = static_residual_programmed(programmed_rdms, ...
    neural_rdms,"paddleOnly",paddleOnly_rdms,corrtype,rsaStruct);

rsaStruct = static_residual_programmed(programmed_rdms, ...
    neural_rdms,"all",{nextTimePoint_rdms paddleOnly_rdms},corrtype,rsaStruct);


%% Run ML-style RNN Residuals

staticRSA_nextTP = static_residual_blackbox( ...
    nextTimePoint_rdms, neural_rdms,"prnn",programmed_rdms,corrtype);

staticRSA_nextTP = static_residual_blackbox( ...
    nextTimePoint_rdms, neural_rdms,"paddleOnly",paddleOnly_rdms,corrtype,staticRSA_nextTP);

staticRSA_nextTP = static_residual_blackbox( ...
    nextTimePoint_rdms, neural_rdms,"all",{programmed_rdms paddleOnly_rdms},corrtype,staticRSA_nextTP);

staticRSA_paddle = static_residual_blackbox( ...
    paddleOnly_rdms, neural_rdms,"prnn",programmed_rdms,corrtype);


%% Helper Functions

function staticRSA = static_residual_programmed(rdm_prnn, rdm_brain,losstype,regressors,corrtype,staticRSA)
%%  

    if ~exist("staticRSA","var"),staticRSA(1:2)=struct;end

    nmodels  = numel(rdm_prnn); 
    dBrain   = zscore(rdm_brain);

    if strcmp(losstype , "all")
        dReg1    = zscore(vertcat(regressors{1}{:})');
        dReg2    = zscore(vertcat(regressors{2}{:})');
        fcall1   = @(x) [dReg1(:,randi(size(dReg1,2))) dReg2(:,randi(size(dReg2,2)))];
        fcallAll = @(x) [dReg1 dReg2];
    else
        nbbrnns  = numel(regressors);
        dRegress = zscore(vertcat(regressors{:})');
        fcall1   = @(x) dRegress(:,randi(nbbrnns));
        fcallAll = dRegress;
    end

    dBrain_all = regress_out(dBrain, fcallAll());
    dBrain_one = zeros(numel(dBrain_all),nmodels);
    dModel_one = dBrain_one;
    dModel_all = dBrain_one;
    dModel_raw = cellfun(@zscore,rdm_prnn,UniformOutput=false);
    for i = 1:nmodels
        printloop(i,nmodels)
        dBrain_one(:,i) = regress_out(dBrain, fcall1());   
        dModel_one(:,i) = regress_out(dModel_raw{i}, fcall1());
        dModel_all(:,i) = regress_out(dModel_raw{i}, fcallAll());
    end
    
    rawTrig = ~strcmp(string(fieldnames(staticRSA)),"raw");
    parfor i = 1:nmodels

        printloop(i,nmodels)
        dModel   = dModel_raw{i}; 
    
        rsa_static_raw   = corr(dBrain', dModel', 'type',corrtype);
        rsa_static_one   = corr(dBrain_one(:,i), dModel_one(:,i), 'type',corrtype);
        rsa_static_all   = corr(dBrain_all, dModel_all(:,i), 'type',corrtype);
                
        strid    = "brain_minus_"+losstype;
        staticRSA(i).(strid+"_residual_oneRegressor")   =  rsa_static_one ;                       
        staticRSA(i).(strid+"_residual_allRegressor")   =  rsa_static_all ; 
        
        if rawTrig,staticRSA(i).("raw")   =  rsa_static_raw ; end

    end

end

function staticRSA = static_residual_blackbox(rdm_bbrnn, rdm_brain,losstype,regressors,corrtype,staticRSA)
%%  

    if ~exist("staticRSA","var"),staticRSA(1:2)=struct;end

    % nprnns  = numel(regressors);
    % dRegress = zscore(vertcat(regressors{:})');

    nmodels  = numel(rdm_bbrnn); 
    dBrain   = zscore(rdm_brain);

    if strcmp(losstype , "all")
        dReg1    = zscore(vertcat(regressors{1}{:})');
        dReg2    = zscore(vertcat(regressors{2}{:})');
        fcall1   = @(x) [dReg1(:,randi(size(dReg1,2))) dReg2(:,randi(size(dReg2,2)))];
        fcallAll = @(x) [dReg1 dReg2];
    else
        nregs  = numel(regressors);
        dRegress = zscore(vertcat(regressors{:})');
        fcall1   = @(x) dRegress(:,randi(nregs));
        fcallAll = dRegress;
    end


    dBrain_all = regress_out(dBrain, fcallAll());
    dBrain_one = zeros(numel(dBrain_all),nmodels);
    dModel_one = dBrain_one;
    dModel_all = dBrain_one;
    dModel_raw = cellfun(@zscore,rdm_bbrnn,UniformOutput=false);
    for i = 1:nmodels
        printloop(i,nmodels)
        dBrain_one(:,i) = regress_out(dBrain, fcall1());   
        dModel_one(:,i) = regress_out(dModel_raw{i}, fcall1());
        dModel_all(:,i) = regress_out(dModel_raw{i}, fcallAll());
    end

    % dBrain_all = regress_out(dBrain, dRegress);
    % dBrain_one = zeros(numel(dBrain_all),nmodels);
    % dModel_one = dBrain_one;
    % dModel_all = dBrain_one;
    % dModel_raw = cellfun(@zscore,rdm_bbrnn,UniformOutput=false);
    % for i = 1:nmodels
    %     printloop(i,nmodels)
    %     reg1 = dRegress(:,randi(nprnns));
    %     dBrain_one(:,i) = regress_out(dBrain, reg1);   
    %     dModel_one(:,i) = regress_out(dModel_raw{i}, reg1);
    %     dModel_all(:,i) = regress_out(dModel_raw{i}, dRegress);
    % end
    
    rawTrig = isempty(fieldnames(staticRSA));
    parfor i = 1:nmodels

        printloop(i,nmodels)
    
        rsa_static_raw   = corr(rdm_brain', rdm_bbrnn{i}', 'type',corrtype);
        rsa_static_one   = corr(dBrain_one(:,i), dModel_one(:,i), 'type',corrtype);
        rsa_static_all   = corr(dBrain_all, dModel_all(:,i), 'type',corrtype);
                
        strid    = "brain_minus_"+losstype;
        staticRSA(i).(strid+"_residual_oneRegressor")   =  rsa_static_one ;                       
        staticRSA(i).(strid+"_residual_allRegressor")   =  rsa_static_all ; 
        
        if rawTrig, staticRSA(i).("raw") =  rsa_static_raw ; end

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









