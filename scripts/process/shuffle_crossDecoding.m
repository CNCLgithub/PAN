%% Suffled decoding

%% Load Neural Data

[bdataNI,bdataNT0,bdataNTE] = get_neural_data(brain);


%% Load PAN models

if 0
    panFile = "../../hpc-outputs/programmed-networks/28-Feb-2025_1000neurons_44a04e02-6fc5-4389-9109-584de368715b/rnn-states/prnn-states-id_49_05630dcb-2eec-4830-b764-e059d6a56bc3.mat";
    PANdata = load(panFile);
    prnn = PANdata.prnn_validation.run_network();
else

    binned_states = cell(3,1);

    run = "../../hpc-outputs/programmed-networks/28-Feb-2025_300neurons_*/rnn-states/";
    binned_states{1} = loadPAN_AllBinnedStates(run);

    run = "../../hpc-outputs/programmed-networks/28-Feb-2025_500neurons_*/rnn-states/";
    binned_states{2} = loadPAN_AllBinnedStates(run);

    run = "../../hpc-outputs/programmed-networks/28-Feb-2025_1000neurons_*/rnn-states/";
    binned_states{3} = loadPAN_AllBinnedStates(run);

end

%% Run Oracle Model

load("../../data/brain_and_rishi_data.mat")

time_bins_sample = 50:50:1000;
nsplits = 1000;
[oracle_corrs , oracle_errors] = run_oracle_data(rishi_data,time_bins_sample,nsplits);

%% Run to Neural Data

[neural_corrs, neural_errors] = run_neural_data(bdataNT0,rishi_data);


%% Run PAN Decoding Data 
  
[pan_corrs_models, pan_errors_models] = deal(cell(3,1));



parfor i = 1:nmodels

    [pan_corrs_all{i}, pan_errors_all{i}] = get_PAN_data_all( ...
        binned_states{j}{i},rishi_data);

end

pan_corrs_models{j}  = pan_corrs_all;
pan_errors_models{j} = pan_errors_all;




pan_corr_avg = cell(3,1);
pan_corr_std = cell(3,1);
for i= 1:3
    pan_corr_avg{i} = squeeze(mean(cat(4,pan_corrs_models{i}{:}), [1 4]))';
    pan_corr_std{i} = squeeze(std(cat(4,pan_corrs_models{i}{:}),[], [1 4]))';
end
pan_corr_avg = permute(cat(3,pan_corr_avg{:}),[1 3 2]);
pan_corr_std = permute(cat(3,pan_corr_std{:}),[1 3 2]);


%% Plot with only PAN 1000 (seperate)

hues = [60 0 180]'/360;
sats = ones(numel(hues),1);
vals = ones(numel(hues),1).*[.8 1 1]';

colors = cat(3,hsvf(hues,sats,vals),hsvf(hues+.05,sats,vals*.8));

ntpts = numel(time_bins_sample);
faces = (1:ntpts*2);
xdata = (faces(1:end/2)'-1)*50;

subplots = {1:3 , 4:6 , 7:8, 9:10,11:12 };

figure

% plot Oracle Data
oracle_avgs   = squeeze(mean(oracle_corrs,[1 2]));
oracle_stdevs = squeeze(std(oracle_corrs, [],[1 2]))./...
    sqrt(sum(size(oracle_corrs,[1 2])));

mdl_avg_bounce0 = oracle_avgs;
mdl_std_bounce0 = oracle_stdevs;

verts0 = mdl_avg_bounce0+[-1 +1].*mdl_std_bounce0;
verts0 = reshape([verts0(:,1),flip(verts0(:,2))],[],1);
verts0 = [[xdata;flip(xdata)],verts0];


subplot(311),hold on
patch("Faces",faces,"Vertices",verts0, ...
    'FaceColor',colors(1,:,1),'FaceAlpha',.2)
plot(xdata , mdl_avg_bounce0,'Color', colors(1,:,1))

title("Oracle")

% plot Neural Data
neural_avgs   = squeeze(mean(neural_corrs,[1 2]));
neural_stdevs = squeeze(std(neural_corrs, [],[1 2]))./...
    sqrt(sum(size(neural_corrs,[1 2])));

mdl_avg_bounce0 = neural_avgs;
mdl_std_bounce0 = neural_stdevs;

verts0 = mdl_avg_bounce0+[-1 +1].*mdl_std_bounce0;
verts0 = reshape([verts0(:,1),flip(verts0(:,2))],[],1);
verts0 = [[xdata;flip(xdata)],verts0];

subplot(subplot(312)),hold on
patch("Faces",faces,"Vertices",verts0, ...
    'FaceColor',colors(2,:,2),'FaceAlpha',.2, ...
    'AlphaDataMapping','none')
plot(xdata , mdl_avg_bounce0,'Color', colors(2,:,2))


title("Neural Data")

% plot PAN Data
strs = string([300 500 1000]);

pan_corr_all = cat(4,pan_corrs_models{i}{:});
mdl_avg_bounce0 = squeeze(mean(pan_corr_all,[1 2 4]));
mdl_std_bounce0 = squeeze(std(pan_corr_all,[],[1 2 4]))./...
    sqrt(sum(size(neural_corrs,[1 2 4])));


verts0 = mdl_avg_bounce0+[-1 +1].*mdl_std_bounce0;
verts0 = reshape([verts0(:,1),flip(verts0(:,2))],[],1);
verts0 = [[xdata;flip(xdata)],verts0];

subplot(313),hold on
patch("Faces",faces,"Vertices",verts0, ...
    'FaceColor',colors(i,:,1),'FaceAlpha',.2)
plot(xdata , mdl_avg_bounce0,'Color', colors(i,:,1))



title(sprintf("PAN %s Latent Units",strs(i)))
sgtitle("Cross Condition Decoding Average (train 0/1-bounce, test 1/0-bounce)")


linkaxes(findall(gcf,'type','axes'))
xlim([250 1000])
ylim([-.1 1])


%% Plot with only PAN 1000 (together)

hues = [60 0 180]'/360;
sats = ones(numel(hues),1);
vals = ones(numel(hues),1).*[.8 1 1]';

colors = cat(3,hsvf(hues,sats,vals),hsvf(hues+.05,sats,vals*.8));

ntpts = numel(time_bins_sample);
faces = (1:ntpts*2);
xdata = (faces(1:end/2)'-1)*50;

subplots = {1:3 , 4:6 , 7:8, 9:10,11:12 };

figure
p = gobjects(3,1);

% plot Oracle Data
oracle_avgs   = squeeze(mean(oracle_corrs,[1 2]));
oracle_stdevs = squeeze(std(oracle_corrs, [],[1 2]))./...
    sqrt(sum(size(oracle_corrs,[1 2])));

mdl_avg_bounce0 = oracle_avgs;
mdl_std_bounce0 = oracle_stdevs;

verts0 = mdl_avg_bounce0+[-1 +1].*mdl_std_bounce0;
verts0 = reshape([verts0(:,1),flip(verts0(:,2))],[],1);
verts0 = [[xdata;flip(xdata)],verts0];


gca,hold on
patch("Faces",faces,"Vertices",verts0, ...
    'FaceColor',colors(1,:,1),'FaceAlpha',.2)
p(1)=plot(xdata , mdl_avg_bounce0,'Color', colors(1,:,1));

% plot Neural Data
neural_avgs   = squeeze(mean(neural_corrs,[1 2]));
neural_stdevs = squeeze(std(neural_corrs, [],[1 2]))./...
    sqrt(sum(size(neural_corrs,[1 2])));

mdl_avg_bounce0 = neural_avgs;
mdl_std_bounce0 = neural_stdevs;

verts0 = mdl_avg_bounce0+[-1 +1].*mdl_std_bounce0;
verts0 = reshape([verts0(:,1),flip(verts0(:,2))],[],1);
verts0 = [[xdata;flip(xdata)],verts0];


patch("Faces",faces,"Vertices",verts0, ...
    'FaceColor',colors(2,:,2),'FaceAlpha',.2, ...
    'AlphaDataMapping','none')
p(2) = plot(xdata , mdl_avg_bounce0,'Color', colors(2,:,2));


% plot PAN Data
strs = string([300 500 1000]);

pan_corr_all = cat(4,pan_corrs_models{3}{:});
mdl_avg_bounce0 = squeeze(mean(pan_corr_all,[1 2 4]));
mdl_std_bounce0 = squeeze(std(pan_corr_all,[],[1 2 4]))./...
    sqrt(sum(size(neural_corrs,[1 2 4])));


verts0 = mdl_avg_bounce0+[-1 +1].*mdl_std_bounce0;
verts0 = reshape([verts0(:,1),flip(verts0(:,2))],[],1);
verts0 = [[xdata;flip(xdata)],verts0];

patch("Faces",faces,"Vertices",verts0, ...
    'FaceColor',colors(3,:,1),'FaceAlpha',.2)
p(3) = plot(xdata , mdl_avg_bounce0,'Color', colors(3,:,1));

legend(p,["Linear Map" "Neural Data" sprintf("PAN %s Latent Units",strs(3))])

title("Trial-Suffled Decoding Average")

xlim([250 1000])
ylim([-.3 1])

%% Load Data Functions

function [bdataNI,bdataNT0,bdataNTE] = get_neural_data(brain)

brainData  = squeeze(num2cell(permute(brain.data,[1 3 2]),[1 2]));

allmask = num2cell(logical(brain.all_mask),2);
ntpts   = cellfun(@(x) find(x,1,"last"),allmask);
tstart  = find(allmask{1},1,"first");

bdata      = cellfun(@(x,idx) x(:,idx),brainData,allmask,UniformOutput=0);
% bdata_norm = cellfun(@(x) zscore(x,[],2),bdata,UniformOutput=0);
bdata_norm = bdata;

ntpts_trim = cellfun(@(x) size(x,2),bdata_norm);

maxpm = 1000/50; % 1s divided by 50ms time bins

maxT  = max(ntpts_trim);
minT  = min(ntpts_trim);

fresamp = @(x) linspace(1,size(x,2),maxT);

bdata_norm_interp = cellfun(@(x) interp1(1:size(x,2),x',fresamp(x))',...
    bdata_norm,UniformOutput=0);

bdata_norm_abs_t0 = cellfun(@(x) x(:,1:maxpm),...
    bdata_norm,UniformOutput=0);

bdata_norm_abs_tend = cellfun(@(x) x(:,(end-maxpm+1):end),...
    bdata_norm,UniformOutput=0);

bdataNI  = zscore(cat(3, bdata_norm_interp{:}),[],[2 ]); % relative time 
bdataNT0 = zscore(cat(3, bdata_norm_abs_t0{:}),[],[2 ]); % absolute time t0 to 1s
bdataNTE = zscore(cat(3, bdata_norm_abs_tend{:}),[],[2 ]); % absolute time tf-1s to tf


end

function binned_states = loadPAN_AllBinnedStates(run)


    files = dir(fullfile(run,"*.mat"));
    nfiles = numel(files);

    binned_states = cell(nfiles,1);

    for i = 1:nfiles

        panFile = fullfile(files(i).folder,files(i).name);
        PANdata = load(panFile);
        prnn = PANdata.prnn_validation.run_network();
        
        maxTbin = 1000/50;
        PANstates = cat(3,prnn.binned_states_raw.modelStates{:});
        PANstates(:,[1:4 maxTbin+5:end],:) = [];
        
        PANstates = zscore(PANstates,[],2);

        binned_states{i} = PANstates;

    end

end

%% Run Functions

function [oracle_corrs , oracle_errors] = run_oracle_data(rishi_data,time_bins_sample,nsplits)

bounce_vec = rishi_data.nbounces;
bounce1 = bounce_vec==1;
bounce0 = ~bounce1;

rd   = rishi_data;
xpos = cellfun(@(x) x(time_bins_sample),rd.sim_coordinates.x,UniformOutput=0);
ypos = cellfun(@(y) y(time_bins_sample),rd.sim_coordinates.y,UniformOutput=0);
xpos = cat(2,xpos{:})';
ypos = cat(2,ypos{:})';

ntpts = numel(time_bins_sample);

ics = cat(3,xpos,ypos,repmat(rd.xdot0,1,ntpts),repmat(rd.ydot0,1,ntpts));
target   = rd.ball_pos_final;  % Nx1 (just y-position)
shuffidx = randperm(numel(target));
target   = target(shuffidx);

corrs  = zeros(nsplits,2);
errors = zeros(nsplits,2);

    for j = 1:ntpts
    
        
        oracle_states = squeeze(ics(:,j,:))';
        
        states_bounce0 = oracle_states(:,bounce0);
        states_bounce1 = oracle_states(:,bounce1); 
        
        % Separate data based on bounce conditions
        target_bounce1 = target(bounce1);
        target_bounce0 = target(bounce0);
        
        nbounce0 = sum(bounce0);
        nbounce1 = sum(bounce1);
        
        ntrain = .8;
        
        Ntr0 = nbounce1;
        Ntr1 = floor(nbounce1*ntrain);
          
        for i = 1:nsplits
        
            %---------- train 0-bounce test 1-bounce
            permidx  = randperm(nbounce0);
            trainidx = permidx(1:Ntr0);
            
            % Solve: ics * M = target (M is 4x1)
            M = states_bounce0(:,trainidx)'\target_bounce0(trainidx);
            
            % Predict on test set
            predicted = states_bounce1' * M;
            
            % Evaluate
            corrs(i,1,j)  = corr(predicted, target_bounce1);
            errors(i,1,j) = norm(predicted - target_bounce1);
        
            %---------- train 1-bounce test 0-bounce
            permidx  = randperm(nbounce1);
            trainidx = permidx(1:Ntr1);
        
            testidx  = randperm(nbounce0);
            testidx((Ntr1+1):end) = [];
            
            % Solve: ics * M = target (M is 4x1)
            M = states_bounce1(:,trainidx)'\target_bounce1(trainidx);
            
            % Predict on test set
            predicted = states_bounce0(:,testidx)' * M;
            
            % Evaluate
            corrs(i,2,j)  = corr(predicted, target_bounce0(testidx));
            errors(i,2,j) = norm(predicted - target_bounce0(testidx));
        
        
        end
        
    
    
    end
    
    oracle_corrs  = corrs;
    oracle_errors = errors;
end

function [neural_corrs, neural_errors] = run_neural_data(bdataNT0,rishi_data)
    bounce_vec = rishi_data.nbounces;
    bounce1 = bounce_vec==1;
    bounce0 = ~bounce1;
    
    rd  = rishi_data;
    target   = rd.ball_pos_final;  % Nx1 (just y-position)
    shuffidx = randperm(numel(target));
    target   = target(shuffidx);

    
    time_bins_sample = 50:50:1000;
    nsamples = numel(time_bins_sample);
    
    nsplits = 1000;
    
    corrs  = zeros(nsplits,2,nsamples);
    errors = zeros(nsplits,2,nsamples);
    
    for j = 1:nsamples
    
    
        time_point = time_bins_sample(j)/50; % (time ms)/(time-bin width)
        
        neural_states = squeeze(bdataNT0(:,time_point,:));
        
        states_bounce0 = neural_states(:,bounce0);
        states_bounce1 = neural_states(:,bounce1); 
        
        % Separate data based on bounce conditions
        target_bounce1 = target(bounce1);
        target_bounce0 = target(bounce0);
        
        nbounce0 = sum(bounce0);
        nbounce1 = sum(bounce1);
        
        ntrain = .8;
        
        Ntr0 = nbounce1;
        Ntr1 = floor(nbounce1*ntrain);
          
        for i = 1:nsplits
        
            %---------- train 0-bounce test 1-bounce
            permidx  = randperm(nbounce0);
            trainidx = permidx(1:Ntr0);
            
            % Solve: ics * M = target (M is 4x1)
            M = states_bounce0(:,trainidx)'\target_bounce0(trainidx);
            
            % Predict on test set
            predicted = states_bounce1' * M;
            
            % Evaluate
            corrs(i,1,j)  = corr(predicted, target_bounce1);
            errors(i,1,j) = norm(predicted - target_bounce1);
        
            %---------- train 1-bounce test 0-bounce
            permidx  = randperm(nbounce1);
            trainidx = permidx(1:Ntr1);
        
            testidx  = randperm(nbounce0);
            testidx((Ntr1+1):end) = [];
            
            % Solve: ics * M = target (M is 4x1)
            M = states_bounce1(:,trainidx)'\target_bounce1(trainidx);
            
            % Predict on test set
            predicted = states_bounce0(:,testidx)' * M;
            
            % Evaluate
            corrs(i,2,j)  = corr(predicted, target_bounce0(testidx));
            errors(i,2,j) = norm(predicted - target_bounce0(testidx));
        
        
        end
        
    
    
    end
    
    neural_corrs  = corrs;
    neural_errors = errors;

end

function [pan_corrs, pan_errors] = get_PAN_data_all(prnnStates,rishi_data)
bounce_vec = rishi_data.nbounces;
bounce1 = bounce_vec==1;
bounce0 = ~bounce1;

rd  = rishi_data;
target   = rd.ball_pos_final;  % Nx1 (just y-position)
shuffidx = randperm(numel(target));
target   = target(shuffidx);

time_bins_sample = 50:50:1000;
nsamples = numel(time_bins_sample);

nsplits = 1000;

corrs  = zeros(nsplits,2,nsamples);
errors = zeros(nsplits,2,nsamples);

PANstates = prnnStates;

for j = 1:nsamples


    time_point = time_bins_sample(j)/50; % (time ms)/(time-bin width)
    
    neural_states = squeeze(PANstates(:,time_point,:));
    
    states_bounce0 = neural_states(:,bounce0);
    states_bounce1 = neural_states(:,bounce1); 
    
    % Separate data based on bounce conditions
    target_bounce1 = target(bounce1);
    target_bounce0 = target(bounce0);
    
    nbounce0 = sum(bounce0);
    nbounce1 = sum(bounce1);
    
    ntrain = .8;
    
    Ntr0 = nbounce1;
    Ntr1 = floor(nbounce1*ntrain);
    
    
    for i = 1:nsplits
    
        %---------- train 0-bounce test 1-bounce
        permidx  = randperm(nbounce0);
        trainidx = permidx(1:Ntr0);
        
        % Solve: ics * M = target (M is 4x1)
        M = states_bounce0(:,trainidx)'\target_bounce0(trainidx);
        
        % Predict on test set
        predicted = states_bounce1' * M;
        
        % Evaluate
        corrs(i,1,j)  = corr(predicted, target_bounce1);
        errors(i,1,j) = norm(predicted - target_bounce1);
    
        %---------- train 1-bounce test 0-bounce
        permidx  = randperm(nbounce1);
        trainidx = permidx(1:Ntr1);
    
        testidx  = randperm(nbounce0);
        testidx((Ntr1+1):end) = [];
        
        % Solve: ics * M = target (M is 4x1)
        M = states_bounce1(:,trainidx)'\target_bounce1(trainidx);
        
        % Predict on test set
        predicted = states_bounce0(:,testidx)' * M;
        
        % Evaluate
        corrs(i,2,j)  = corr(predicted, target_bounce0(testidx));
        errors(i,2,j) = norm(predicted - target_bounce0(testidx));
    
    
    end
    


end

pan_corrs  = corrs;
pan_errors = errors;

end









