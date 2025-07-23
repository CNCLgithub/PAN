%% RSA Pipeline (Cleaned and Commented)
% Author: Daniel Calbick
% Date: 2025-05-02
% Description: Full RSA pipeline for comparing brain, ML-style, 
% and programmed networks.

%% -------------------------------------------------------------------------
%                             Load Programmed Data
%% -------------------------------------------------------------------------
output_dir = "../../hpc-outputs/programmed-networks";
runid      = "28-Feb-2025_1000neurons_44a04e02-6fc5-4389-9109-584de368715b";

pth     = fullfile(output_dir, runid);
statedir = "rnn-states";
svdir   = "rnn-analysis";
svname  = "rdm-vectors.mat";

if ~exist("brain", "var")
    load("../../data/brain_and_rishi_data.mat", "brain")
end

if exist(fullfile(pth, svdir, svname), "file")
    load(fullfile(pth, svdir, svname))
else
    files   = dir(fullfile(pth, "*.mat"));
    nfiles  = numel(files); 
    progMat = cell(nfiles, 1);

    mask = brain.all_mask(:, 7:end);
    tpts = cellfun(@(s) find(s, 1, "last"), num2cell(mask, 2));

    for i = 1:nfiles
        a = load(fullfile(pth, files(i).name), "prnn_validation");
        prnn = a.prnn_validation;
        prnn.run_network;

        states = cellfun(@(x) x(:, 5:end), prnn.binned_states.modelStates, 'UniformOutput', false);
        ntrials = numel(states);

        stateCell = cell(ntrials, 1);
        for j = 1:ntrials
            stateCell{j} = zscore(states{j}(:, 1:tpts(j)), [], 2);
        end

        progMat{i} = cat(2, stateCell{:});
    end

    rdmProg = cellfun(@(s) pdist(s', "euclidean"), progMat, 'UniformOutput', false);

    if ~exist(fullfile(pth, svdir), 'dir'), mkdir(fullfile(pth, svdir)); end
    save(fullfile(pth, svdir, "state-matrices.mat"), "progMat")
    save(fullfile(pth, svdir, svname), "rdmProg", "-v7.3")
end

%% -------------------------------------------------------------------------
%                        Load Brain & Black Box Data
%% -------------------------------------------------------------------------
load("../../data/bbrnns.mat", "bbrnns")

if exist("../../data/staticRDMs_brain_bbrnn.mat", "file")
    load("../../data/staticRDMs_brain_bbrnn.mat")
else
    braindata = zscore(brain.data(:, :, 7:end-1), [], 3);
    brainData = squeeze_cell(braindata);

    bbData = cellfun(@(s) squeeze_cell(zscore(double(s), [], 3)), {bbrnns.data_neur_nxcxt}', 'UniformOutput', false);
    bbData = cat(1, bbData{:});

    mask = brain.all_mask(:, 7:end);
    tpts = cellfun(@(s) find(s, 1, "last"), num2cell(mask, 2));
    ntrials = size(mask, 1);

    brainCell = cell(ntrials, 1);
    bbCell = cell(ntrials, 1);

    for i = 1:ntrials
        brainCell{i} = brainData{i}(:, 1:tpts(i));
        bbCell{i} = cellfun(@(s) s(:, 1:tpts(i)), bbData(:, i), 'UniformOutput', false);
    end

    bbCell = cat(2, bbCell{:});
    bbMats = cellfun(@(s) horzcat(s{:}), num2cell(bbCell, 2), 'UniformOutput', false);
    brainMat = cat(2, brainCell{:});

    rdmBBs = cellfun(@(s) pdist(s', "euclidean"), bbMats, 'UniformOutput', false);
    rdmBrain = pdist(brainMat', "euclidean");

    save("../../data/staticRDMs_brain_bbrnn.mat", "rdmBBs", "rdmBrain", '-v7.3')
end

%% -------------------------------------------------------------------------
%                        Run Raw Correlations with parfeval
%% -------------------------------------------------------------------------
bbrnncor = asyncCorr(rdmBrain, rdmBBs);
prnncor  = asyncCorr(rdmBrain, rdmProg);

%% -------------------------------------------------------------------------
%                     Plot Raw RSA Correlations Grouped
%% -------------------------------------------------------------------------
plotGroupedRawCorrelations(bbrnncor, prnncor, bbrnns);

%% -------------------------------------------------------------------------
%                    Run Residual RSA for Programmed Models
%% -------------------------------------------------------------------------

forceRun1 = false;

if ~exist("rsaStruct","var") || forceRun1
    % Define logical keys for model grouping
    losskey = contains(string({bbrnns.loss_type}), "-mov");
    neurkey = cellfun(@(s) size(s,1), {bbrnns.data_neur_nxcxt}) == 40;
    inpkey  = contains(string({bbrnns.input_representation}), "gabor");
    
    % Group indices
    nextp   = losskey & neurkey & inpkey;
    pdlonly = ~losskey & neurkey & inpkey;
    
    % Assign RDM sets
    nextTimePoint_rdms = rdmBBs(nextp);
    paddleOnly_rdms    = rdmBBs(pdlonly);
    programmed_rdms    = rdmProg;
    neural_rdms        = rdmBrain;
    
    corrtype = "Pearson";
    
    % Run residual RSA
    rsaStruct = run_time_collapsed_residual_programmed(programmed_rdms, neural_rdms, "nextTimePoint", nextTimePoint_rdms, corrtype);
    rsaStruct = run_time_collapsed_residual_programmed(programmed_rdms, neural_rdms, "paddleOnly", paddleOnly_rdms, corrtype, rsaStruct);
    rsaStruct = run_time_collapsed_residual_programmed(programmed_rdms, neural_rdms, "all", {nextTimePoint_rdms paddleOnly_rdms}, corrtype, rsaStruct);
end

% Plot programmed residuals
plotResidualRSAProgrammed(rsaStruct);

%% -------------------------------------------------------------------------
%                    Run Residual RSA for Black Box Models
%% -------------------------------------------------------------------------

forceRun2 = false;

if ~exist("staticRSA_nextTP","var") || forceRun2
    staticRSA_nextTP = run_time_collapsed_residual_blackbox(nextTimePoint_rdms, neural_rdms, "prnn", programmed_rdms, corrtype);
    staticRSA_nextTP = run_time_collapsed_residual_blackbox(nextTimePoint_rdms, neural_rdms, "paddleOnly", paddleOnly_rdms, corrtype, staticRSA_nextTP);
    staticRSA_paddle = run_time_collapsed_residual_blackbox(paddleOnly_rdms, neural_rdms, "prnn", programmed_rdms, corrtype);
end

% Plot black box residuals
plotResidualRSABlackBox(staticRSA_nextTP, staticRSA_paddle);

%%

% close all
% clc

% Plot violin comparison with significance (Programmed vs. NextTP)
plotViolinComparisonRSA(rsaStruct, staticRSA_nextTP);

%%


%% -------------------------------------------------------------------------
%                    Local Plotting Functions
%% -------------------------------------------------------------------------

function plotViolinComparisonRSA(rsaStruct, staticRSA_nextTP)
% ---------------------------------------------------------------
% Seven-way violin plot (built-in violinplot)
% Order: Prog-Raw • Prog-NextTP • Prog-PaddleOnly • Prog-Both •
%        NextTP-Raw • NextTP-PaddleOnly • NextTP-Prog
% ---------------------------------------------------------------

% ---------- 1.  Collect column-vectors -------------------------
P = vertcat(rsaStruct);    % programmed model struct array

vals = { ...
    double([P.raw]),                                             ... % Prog-Raw
    double([P.brain_minus_nextTimePoint_residual_allRegressor]), ... % Prog-NextTP
    double([P.brain_minus_paddleOnly_residual_allRegressor]),    ... % Prog-Pdl
    double([P.brain_minus_all_residual_allRegressor]),           ... % Prog-Both
    double([staticRSA_nextTP.raw]),                              ... % NextTP-Raw
    double([staticRSA_nextTP.brain_minus_paddleOnly_residual_allRegressor]), ... % NextTP-Pdl
    double([staticRSA_nextTP.brain_minus_prnn_residual_allRegressor]),... % NextTP-Prog
    double([staticRSA_nextTP.brain_minus_all_residual_allRegressor])};      

% Common length (shorter vectors keep all rows, longer ones are clipped)
N    = min(cellfun(@numel, vals));
vals = cellfun(@(v)v(1:N)', vals, 'uni', 0);   % force row-vectors

% ---------- 2.  Long-form arrays for built-in violinplot -------
labels = {'Prog-Raw','Prog-NextTP','Prog-PaddleOnly','Prog-Both', ...
          'NextTP-Raw','NextTP-PaddleOnly','NextTP-Prog','NextTP-Both'};

y = vertcat(vals{:});          % numeric column
grp = repelem(labels, N).';      % cell array of char vectors
grp = categorical(grp, labels, 'Ordinal', true);  % preserves left→right order

% ---------- 3.  Plot -------------------------------------------
figure;
v = violinplot(grp,y , "DensityScale","count");
ylabel('RSA (Pearson / Spearman)');
title('RSA residuals (-one regressors): Programmed vs NextTP');
ylim([0 1]); grid on;

% ---------- 4.  Optional paired-star annotations ---------------
pairs = [1 5; 2 7; 3 6];          % Raw, NextTP-resid, Paddle-resid comparisons
addStars(gca, reshape(y, N, []), pairs);  % your existing helper

fx = @(y) randn(numel(y),1)*0.1;

for i = 1:numel(vals)
    scatter(fx(vals{i})+i,vals{i});
end



end

function addStars(axHandle, data, pairs)
    nPair  = size(pairs,1);
    p_data = zeros(nPair,1);

    for p = 1:nPair
        i = pairs(p,1); j = pairs(p,2);
        [~, p_data(p)] = ttest2(data(:,i), data(:,j), 'Vartype','unequal');
    end

    starfun = @(p) ...
        repmat('*',1, 3*(p<0.001) + 2*(p>=0.001 & p<0.01) + ...
                      1*(p>=0.01 & p<0.05) );  % '', *, **, ***
    stars = arrayfun(starfun, p_data, 'uni', 0);

    hold(axHandle, "on");
    yl = ylim; yStart = yl(2) * 1.02;
    yStep = diff(yl) * 0.07;

    for p = 1:nPair
        x1 = pairs(p,1); x2 = pairs(p,2);
        y = yStart + (p-1)*yStep;
        plot(axHandle, [x1 x1 x2 x2], [y-0.01 y y y-0.01], 'k', 'LineWidth', 1)
        text(axHandle, mean([x1 x2]), y+0.01, stars{p}, ...
            'HorizontalAlignment','center','FontSize',12)
    end

    ylim(axHandle, [yl(1) yStart + (nPair-1)*yStep + 0.05])
end


%% -------------------------------------------------------------------------
%                    Local Plotting Functions
%% -------------------------------------------------------------------------

function plotResidualRSAProgrammed(rsaStruct)
    fields = [
        "raw"                    "Raw"
        "brain_minus_nextTimePoint_residual_oneRegressor"  "-nextOne"
        "brain_minus_nextTimePoint_residual_allRegressor"  "-nextAll"
        "brain_minus_paddleOnly_residual_oneRegressor"     "-paddleOne"
        "brain_minus_paddleOnly_residual_allRegressor"     "-paddleAll"
        "brain_minus_all_residual_oneRegressor"            "-bothOne"
        "brain_minus_all_residual_allRegressor"            "-bothAll"
    ];

    nfields = size(fields, 1);
    a = vertcat(rsaStruct);
    fx = @(d) randn(numel(d),1) * .1;

    figure; gca; hold on
    for i = 1:nfields
        data = [a.(fields(i,1))];
        scatter(fx(data) + i, data, 'filled')
    end
    hold off
    ylim([0, 1])
    xticks(1:nfields)
    xticklabels(fields(:,2))
    title('Programmed Residual RSA')
    grid on
end

function plotResidualRSABlackBox(staticRSA_nextTP, staticRSA_paddle)
    % Panel 1 - Next Time Point
    fields1 = [
        "raw"             "Raw"
        "brain_minus_prnn_residual_oneRegressor"         "-prnnOne"
        "brain_minus_prnn_residual_allRegressor"         "-prnnAll"
        "brain_minus_paddleOnly_residual_oneRegressor"   "-paddleOne"
        "brain_minus_paddleOnly_residual_allRegressor"   "-paddleAll"
    ];

    % Panel 2 - Paddle Only
    fields2 = [
        "raw" "Raw"
        "brain_minus_prnn_residual_oneRegressor" "-prnnOne"
        "brain_minus_prnn_residual_allRegressor" "-prnnAll"
    ];

    fx = @(d) randn(numel(d), 1) * .1;

    figure;
    subplot(1,3,1), hold on, title("Next TP Residual RSA")
    for i = 1:size(fields1,1)
        data = [staticRSA_nextTP.(fields1(i,1))];
        scatter(fx(data) + i, data, 'filled')
    end
    hold off, ylim([0,1]), xticks(1:size(fields1,1)), xticklabels(fields1(:,2)), grid on

    subplot(1,3,2), hold on, title("Paddle Only Residual RSA")
    for i = 1:size(fields2,1)
        data = [staticRSA_paddle.(fields2(i,1))];
        scatter(fx(data) + i, data, 'filled')
    end
    hold off, ylim([0,1]), xticks(1:size(fields2,1)), xticklabels(fields2(:,2)), grid on

    subplot(1,3,3), hold on, title("Overlayed RSA Residuals")
    for i = 1:size(fields1,1)
        d1 = [staticRSA_nextTP.(fields1(i,1))];
        d2 = [staticRSA_paddle.(fields1(min(i, size(fields2,1)),1))];
        scatter(fx(d1) + i - 0.15, d1, 'filled')
        scatter(fx(d2) + i + 0.15, d2, 'filled')
    end
    hold off, ylim([0,1]), xticks(1:size(fields1,1)), xticklabels(fields1(:,2)), grid on
end


%% -------------------------------------------------------------------------
%                    Utility Function Definitions (Local)
%% -------------------------------------------------------------------------

function staticRSA = run_time_collapsed_residual_programmed(rdm_prnn, rdm_brain,losstype,regressors,corrtype,staticRSA)
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

function staticRSA = run_time_collapsed_residual_blackbox(rdm_bbrnn, rdm_brain,losstype,regressors,corrtype,staticRSA)
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

function result = asyncCorr(rdmBrain, rdmSet)
    n = numel(rdmSet);
    result = zeros(n, 1);
    futures = parallel.FevalFuture.empty(n, 0);
    pool = gcp();

    for i = 1:n
        futures(i) = parfeval(pool, @computeCorr, 1, rdmBrain, rdmSet{i}, i, n);
    end
    for i = 1:n
        [idx, val] = fetchNext(futures);
        result(idx) = val;
    end
end

function r = computeCorr(rdm1, rdm2, idx, total)
    printloop(idx, total);
    r = corr(rdm1', rdm2', 'Type', 'Spearman');
end

function squeezed = squeeze_cell(data)
    tmp = num2cell(data, [1 3])';
    squeezed = cellfun(@squeeze, tmp, 'UniformOutput', false);
end

function plotGroupedRawCorrelations(bbrnncor, prnncor, bbrnns)
    losskey = contains(string({bbrnns.loss_type}), "-mov");
    neurkey = cellfun(@(s) size(s, 1), {bbrnns.data_neur_nxcxt}) == 40;
    inpkey  = contains(string({bbrnns.input_representation}), "gabor");

    nextp   = losskey & neurkey & inpkey;
    pdlonly = ~losskey & neurkey & inpkey;

    cor_next_tp  = bbrnncor(nextp);
    cor_pdl_only = bbrnncor(pdlonly);

    fx = @(d) randn(numel(d), 1) * 0.1;

    figure;
    scatter(fx(cor_next_tp), cor_next_tp, 'filled'); hold on
    scatter(fx(cor_pdl_only)+1, cor_pdl_only, 'filled');
    scatter(fx(prnncor)+2, prnncor, 'filled');

    xticks(0:2)
    xticklabels({"NextTP", "PaddleOnly", "Programmed"})
    ylabel("Spearman Correlation")
    title("RSA Similarity with Brain RDM")
    ylim([0, 1])
    grid on
    hold off
end
