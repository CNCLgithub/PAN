function calculate_RDMs_for_neural_and_ML_models()
% Calculates and saves RDMs for neural and ML networks. 
%
% ----- Normalizing and Partitioning State Matrices
%
% The script loads the neural data and the ML-style models' data
%
% Then it normalizes them via zscore **during the trial-relevent period**,
% by taking the mean and standard dev within only timebins relevent to the
% stimuli.
%
% We then organize the state matrixes into cell arrays or dimension:
% {ntimebins x nmodels} -> w/ elem. {[nneurons x ntrials]}
%
% These state matrices are then saved to the data/ dir.
%
% ------- Computing RDMs for both static and time-resolved RSA
%
% 
% 
%% Load In Data and Organize Machine-Learning style models from Hansem

    load("../../data/brain_and_rishi_data.mat","brain")
    load("../../data/bbrnns.mat","bbrnns")

    
    % Organize ML-style models into groups
    % We pooling networks {all-sim, all-sim2, and vis-sim} into "nextTP"
    % models and no-sim is the "paddleOnly" model family.
    %
    % Further, we want to only look at the best of the models so we limit
    % the input representation to Gabor-Input and 40 neurons

    nneurons = cellfun(@(x) size(x,1),{bbrnns.data_neur_nxcxt})';
    inp      = string({bbrnns.input_representation})';

    idxKeep = (nneurons == 40) & contains(inp,"gabor");

    analysis_MLmodel_key = struct2table(bbrnns(idxKeep,:));

    loss         = string(analysis_MLmodel_key.loss_type)'; 
    lossGroupIdx = contains(loss,"-");

    nML = sum(idxKeep);

    analysisGroup = repmat("paddleOnly",nML,1);
    analysisGroup(lossGroupIdx) = "nextTP";

    analysis_MLmodel_key.analysisGroup = analysisGroup;

    %%

    mask       = num2cell(logical(brain.all_mask),2);
    neuralData = squeeze(num2cell(permute(brain.data,[1 3 2]),[1 2]));

    

    avgf  = @(d,idx) mean(d(:,idx),2);
    stdf  = @(d,idx) std(d(:,idx),[],2);
    normf = @(d,idx) (d - avgf(d,idx))./stdf(d,idx);

    ndata = cellfun(@(d,idx) normf(d,idx), neuralData,mask,UniformOutput=false);
    
    ndata = cat(3,ndata{:});
    ndata(isnan(ndata)) = 0;
    ndata = squeeze(num2cell(ndata,[1 2]));

    

    %%
    
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