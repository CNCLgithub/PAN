%% Test code for oracle

bounce_vec = rishi_data.nbounces;
bounce1 = bounce_vec==1;
bounce0 = ~bounce1;

rd = rishi_data;
ics = [rd.x0 rd.y0 rd.xdot0 rd.ydot0];  % Nx4
target = rd.ball_pos_final;  % Nx1 (just y-position)

nsplits = 1000;

% Separate data based on bounce conditions
ics_bounce1    = ics(bounce1, :);
target_bounce1 = target(bounce1);
ics_bounce0    = ics(bounce0, :);
target_bounce0 = target(bounce0);

nbounce0 = sum(bounce0);
nbounce1 = sum(bounce1);

ntrain = .8;

Ntr0 = floor(nbounce0*ntrain);
Ntr1 = floor(nbounce1*ntrain);

corrs  = zeros(nsplits,2);
errors = zeros(nsplits,2);

for j = 1:ntpts
    for i = 1:nsplits
    
        %---------- bounce0
        permidx  = randperm(nbounce0);
        trainidx = permidx(1:Ntr0);
        testidx  = permidx((Ntr0+1):end);
        
        % Solve: ics * M = target (M is 4x1)
        M = ics_bounce0(trainidx,:)\target_bounce0(trainidx);
        
        % Predict on test set
        predicted = ics_bounce0(testidx,:) * M;
        
        % Evaluate
        corrs(i,1)  = corr(predicted, target_bounce0(testidx));
        errors(i,1) = norm(predicted - target_bounce0(testidx));
    
        %---------- bounce1
        permidx  = randperm(nbounce1);
        trainidx = permidx(1:Ntr1);
        testidx  = permidx((Ntr1+1):end);
        
        % Solve: ics * M = target (M is 4x1)
        M = ics_bounce1(trainidx,:)\target_bounce1(trainidx);
        
        % Predict on test set
        predicted = ics_bounce1(testidx,:) * M;
        
        % Evaluate
        corrs(i,2)  = corr(predicted, target_bounce1(testidx));
        errors(i,2) = norm(predicted - target_bounce1(testidx));
    
    
    end

end
bar(mean(corrs)); hold on
errorbar(1:2,mean(corrs),std(corrs),'k','Linewidth',3,'LineStyle','none')
xticklabels(["No Bounce", "Bounce"])
ylim([0 1])

%% Oracle Train on one test on the other

load("../../data/brain_and_rishi_data.mat")

time_bins_sample = 50:50:1000;
nsplits = 1000;
[oracle_corrs , oracle_errors] = run_oracle_data(rishi_data,time_bins_sample,nsplits);

% plot_oracle_data_time(oracle_corrs,oracle_errors,time_bins_sample)

%% Load Neural Data

[bdataNI,bdataNT0,bdataNTE] = get_neural_data(brain);


%% Extend Run to Neural Data

[neural_corrs, neural_errors] = run_neural_data(bdataNT0,rishi_data);


%% Run with a PAN model(s)

if 0
    %%
    panFile = "../../hpc-outputs/programmed-networks/28-Feb-2025_1000neurons_44a04e02-6fc5-4389-9109-584de368715b/rnn-states/prnn-states-id_49_05630dcb-2eec-4830-b764-e059d6a56bc3.mat";
    PANdata = load(panFile);
    prnn = PANdata.prnn_validation.run_network();
else

    binned_states = cell(3,1);

    if 0
        run = "../../hpc-outputs/programmed-networks/28-Feb-2025_300neurons_*/rnn-states/";
        binned_states{1} = loadPAN_AllBinnedStates(run);
    
        run = "../../hpc-outputs/programmed-networks/28-Feb-2025_500neurons_*/rnn-states/";
        binned_states{2} = loadPAN_AllBinnedStates(run);
    end
    run = "../../hpc-outputs/programmed-networks/28-Feb-2025_1000neurons_*/rnn-states/";
    binned_states{3} = loadPAN_AllBinnedStates(run);

end

%% Extract PAN Decoding Data 

[pan_corrs_models, pan_errors_models] = deal(cell(3,1));

for j = 1:3

    nmodels = numel(binned_states{j});

    [pan_corrs_all, pan_errors_all] = deal(cell(nmodels,1));

    parfor i = 1:nmodels

        [pan_corrs_all{i}, pan_errors_all{i}] = run_PAN_data_all( ...
            binned_states{j}{i},rishi_data);

    end

    pan_corrs_models{j}  = pan_corrs_all;
    pan_errors_models{j} = pan_errors_all;

end

pan_corr_avg = cell(3,1);
pan_corr_std = cell(3,1);
for i= 1:3
    pan_corr_avg{i} = squeeze(mean(cat(4,pan_corrs_models{i}{:}), [1 4]))';
    pan_corr_std{i} = squeeze(std(cat(4,pan_corrs_models{i}{:}),[], [1 4]))';
end
pan_corr_avg = permute(cat(3,pan_corr_avg{:}),[1 3 2]);
pan_corr_std = permute(cat(3,pan_corr_std{:}),[1 3 2]);


%% Plot With all PAN units

hues = [60 0 150 180 220]'/360;
sats = ones(5,1);
vals = ones(5,1);

colors = cat(3,hsvf(hues,sats,vals),hsvf(hues+.05,sats,vals*.8));

% close all
% cimage(colors(:,:,1)),cimage(colors(:,:,2))
ntpts = numel(time_bins_sample);
faces = (1:ntpts*2);
xdata = (faces(1:end/2)'-1)*50;

subplots = {1:3 , 4:6 , 7:8, 9:10,11:12 };

figure

% plot Oracle Data
oracle_avgs   = squeeze(mean(oracle_corrs))';
oracle_stdevs = squeeze(std(oracle_corrs))';

mdl_avg_bounce0 = mean(oracle_avgs,2);
% mdl_avg_bounce1 = oracle_avgs(:,2);

mdl_std_bounce0 = mean(oracle_stdevs,2);
mdl_std_bounce1 = oracle_stdevs(:,2);

verts0 = mdl_avg_bounce0+[-1 +1].*mdl_std_bounce0;
% verts1 = mdl_avg_bounce1+[-1 +1].*mdl_std_bounce1;
verts0 = reshape([verts0(:,1),flip(verts0(:,2))],[],1);
% verts1 = reshape([verts1(:,1),flip(verts1(:,2))],[],1);
verts0 = [[xdata;flip(xdata)],verts0];
% verts1 = [[xdata;flip(xdata)],verts1];

subplot(2,6,subplots{1}),hold on
patch("Faces",faces,"Vertices",verts0, ...
    'FaceColor',colors(1,:,1),'FaceAlpha',.2)
plot(xdata , mdl_avg_bounce0,'Color', colors(1,:,1))

% patch("Faces",faces,"Vertices",verts1, ...
%         'FaceColor',colors(1,:,2),'FaceAlpha',.2, ...
%         'AlphaDataMapping','none')
% plot(xdata , mdl_avg_bounce1,'--','Color', colors(1,:,2))
title("Oracle")

% plot Neural Data

neural_avgs   = squeeze(mean(neural_corrs))';
neural_stdevs = squeeze(std(neural_corrs))';

mdl_avg_bounce0 = mean(neural_avgs(:,1),3);
mdl_avg_bounce1 = neural_avgs(:,2);

mdl_std_bounce0 = mean(neural_stdevs(:,1),3);
mdl_std_bounce1 = neural_stdevs(:,2);

verts0 = mdl_avg_bounce0+[-1 +1].*mdl_std_bounce0;
% verts1 = mdl_avg_bounce1+[-1 +1].*mdl_std_bounce1;
verts0 = reshape([verts0(:,1),flip(verts0(:,2))],[],1);
% verts1 = reshape([verts1(:,1),flip(verts1(:,2))],[],1);
verts0 = [[xdata;flip(xdata)],verts0];
% verts1 = [[xdata;flip(xdata)],verts1];

subplot(2,6,subplots{2}),hold on
patch("Faces",faces,"Vertices",verts0, ...
    'FaceColor',colors(2,:,2),'FaceAlpha',.2, ...
    'AlphaDataMapping','none')
plot(xdata , mdl_avg_bounce0,'Color', colors(2,:,2))
xlim([200 1000])
% patch("Faces",faces,"Vertices",verts1, ...
%         'FaceColor',colors(2,:,2),'FaceAlpha',.2, ...
%         'AlphaDataMapping','none')
% plot(xdata , mdl_avg_bounce1,'--','Color', colors(2,:,2))
title("Neural Data")

% plot PAN Data
strs = string([300 500 1000]);
for i = 1:3
    
    mdl_avg_bounce0 = mean(pan_corr_avg(:,i,1),3);
    % mdl_avg_bounce1 = pan_corr_avg(:,i,2);

    mdl_std_bounce0 = mean(pan_corr_std(:,i,1),3);
    % mdl_std_bounce1 = pan_corr_std(:,i,2);

    verts0 = reshape(mdl_avg_bounce0+[-1 +1].*mdl_std_bounce0,[],1);
    % verts1 = reshape(mdl_avg_bounce1+[-1 +1].*mdl_std_bounce1,[],1);
    verts0 = [[xdata;flip(xdata)],verts0];
    % verts1 = [[xdata;flip(xdata)],verts1];


    subplot(2,6,subplots{2+i}),hold on
    patch("Faces",faces,"Vertices",verts0, ...
        'FaceColor',colors(i+2,:,1),'FaceAlpha',.2)
    plot(xdata , mdl_avg_bounce0,'Color', colors(i+2,:,1))
    

    % patch("Faces",faces,"Vertices",verts1, ...
    %     'FaceColor',colors(i+2,:,2),'FaceAlpha',.2, ...
    %     'AlphaDataMapping','none')
    % plot(xdata , mdl_avg_bounce1,'--','Color', colors(i+2,:,2))

    title(sprintf("PAN %s Latent Units",strs(i)))
    


end

linkaxes(findall(gcf,'type','axes'),'y')

%% Plot with only PAN 1000

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

hues = [60 0 180 180 180]'/360;
sats = ones(numel(hues),1);
vals = ones(numel(hues),1).*[.8 1 .5 .7 1]';

colors = hsvf(hues,sats,vals);

ntpts = numel(time_bins_sample);
faces = (1:ntpts*2);
xdata = (faces(1:end/2)'-1)*50;


figure
p = gobjects(5,1);

% plot Oracle Data
oracle_avgs   = squeeze(mean(oracle_corrs,[1 2]));
oracle_stdevs = squeeze(std(oracle_corrs, [],[1 2]))./...
    sqrt(sum(size(oracle_corrs,[1 2])));

oracle_avg_bounce0 = oracle_avgs;
mdl_std_bounce0 = oracle_stdevs;

verts0 = oracle_avg_bounce0+[-1 +1].*mdl_std_bounce0;
verts0 = reshape([verts0(:,1),flip(verts0(:,2))],[],1);
verts0 = [[xdata;flip(xdata)],verts0];


gca,hold on
patch("Faces",faces,"Vertices",verts0, ...
    'FaceColor',colors(1,:),'FaceAlpha',.2)
p(1)=plot(xdata , oracle_avg_bounce0,'Color', colors(1,:));

% plot Neural Data
neural_avgs   = squeeze(mean(neural_corrs,[1 2]));
neural_stdevs = squeeze(std(neural_corrs, [],[1 2]))./...
    sqrt(sum(size(neural_corrs,[1 2])));

brain_avg_bounce0 = neural_avgs;
mdl_std_bounce0   = neural_stdevs;

verts0 = brain_avg_bounce0+[-1 +1].*mdl_std_bounce0;
verts0 = reshape([verts0(:,1),flip(verts0(:,2))],[],1);
verts0 = [[xdata;flip(xdata)],verts0];


patch("Faces",faces,"Vertices",verts0, ...
    'FaceColor',colors(2,:),'FaceAlpha',.2, ...
    'AlphaDataMapping','none')
p(2) = plot(xdata , brain_avg_bounce0,'Color', colors(2,:));


% plot PAN Data
strs = string([300 500 1000]);
svxdata = cell(3,1);
for i = 1:3
    pan_corr_all = cat(4,pan_corrs_models{(i)}{:});
    mdl_avg_bounce0 = squeeze(mean(pan_corr_all,[1 2 4]));
    mdl_std_bounce0 = squeeze(std(pan_corr_all,[],[1 2 4]))./...
        sqrt(sum(size(pan_corr_all,[1 2 4])));

    sv{i} = mdl_avg_bounce0;
    
    
    verts0 = mdl_avg_bounce0+[-1 +1].*mdl_std_bounce0;
    verts0 = reshape([verts0(:,1),flip(verts0(:,2))],[],1);
    verts0 = [[xdata;flip(xdata)],verts0];
    
    patch("Faces",faces,"Vertices",verts0, ...
        'FaceColor',colors(3,:),'FaceAlpha',.2)
    p(2+i) = plot(xdata , mdl_avg_bounce0,'Color', colors(2+i,:));

end

legend(p,["Linear Map" "Neural Data" ...
    sprintf("PAN %s Latent Units",strs(1)) ...
    sprintf("PAN %s Latent Units",strs(2)) ...
    sprintf("PAN %s Latent Units",strs(3))])

title("Cross Condition Decoding Average" + newline + ...
 "(train 0/1-bounce, test 1/0-bounce)")

xlim([250 1000])
ylim([-.1 1.2])

%%

tbin = 6;
pan1000_avg_corr = sv{3}(6:end);   
neural_avg_corr  = neural_avgs(6:end);  
oracle_avg_corr  = oracle_avgs(6:end);  
fp = @(s1,s2,cor,rho) ...
    fprintf("corr(%s,%s) = %.3f, p = %.3f\n",s1,s2,cor,rho);

[c,p]=corr(pan1000_avg_corr,neural_avg_corr);
fp("neural","PAN",c,p)
[c,p]=corr(oracle_avg_corr,neural_avg_corr);
fp("neural","Linear Map",c,p)
[c,p]=corr(pan1000_avg_corr,oracle_avg_corr);
fp("Linear Map","PAN",c,p)


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

function [pan_corrs, pan_errors] = get_PAN_data(prnn,rishi_data)
bounce_vec = rishi_data.nbounces;
bounce1 = bounce_vec==1;
bounce0 = ~bounce1;

rd  = rishi_data;
target = rd.ball_pos_final;  % Nx1 (just y-position)

time_bins_sample = 50:50:1000;
nsamples = numel(time_bins_sample);

nsplits = 1000;

corrs  = zeros(nsplits,2,nsamples);
errors = zeros(nsplits,2,nsamples);

maxTbin = 1000/50;
PANstates = cat(3,prnn.binned_states_raw.modelStates{:});
PANstates(:,[1:4 maxTbin+5:end],:) = [];

PANstates = zscore(PANstates,[],2);

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

%% Decoding Functions

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
target = rd.ball_pos_final;  % Nx1 (just y-position)

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
    target = rd.ball_pos_final;  % Nx1 (just y-position)
    
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

function [pan_corrs, pan_errors] = run_PAN_data_all(prnnStates,rishi_data)
bounce_vec = rishi_data.nbounces;
bounce1 = bounce_vec==1;
bounce0 = ~bounce1;

rd  = rishi_data;
target = rd.ball_pos_final;  % Nx1 (just y-position)

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


%% Plot Functions

function ax = init_plot_oracle(oracle_corrs,oracle_errors)

figure
ax = gobjects(2,2);

ax(1,1) = subplot(241);
b = bar([[nan nan];mean(oracle_corrs)]); hold on
% errorbar(1:2,mean(corrs)',std(corrs)','k','Linewidth',3,'LineStyle','none')

xpos = vertcat(b.XEndPoints);
errorbar(xpos(:,2),mean(oracle_corrs),std(oracle_corrs),'k','Linewidth',3,'LineStyle','none')

xticks(xpos(:,2))
xticklabels(["0 Bounce", "1 Bounce"])
ylim([0 1])
ylabel(ax(1,1),"Performance")
title("Oracle")
hold off

ax(2,1) = subplot(245);
bar([[nan nan];mean(oracle_errors)]); hold on

errorbar(xpos(:,2),mean(oracle_errors),std(oracle_errors),'k','Linewidth',3,'LineStyle','none')

xticks(xpos(:,2))
xticklabels(["0 Bounce", "1 Bounce"])
ylabel(ax(2,1),"RMSE")
hold off

xlim(ax(:,1),[1.5 2.5])

end

function plot_neural_data(ax,neural_corrs,neural_errors,time_bins_sample)

ax(1,2) = subplot(2,4,2:4);

avgs   = squeeze(mean(neural_corrs))';
stdevs = squeeze(std(neural_corrs))';

b = bar(avgs); hold on
xpos = {b.XEndPoints};

errorbar(xpos{1},avgs(:,1),stdevs(:,1),'k','Linewidth',3,'LineStyle','none')
errorbar(xpos{2},avgs(:,2),stdevs(:,2),'k','Linewidth',3,'LineStyle','none')

title(ax(1,2),"Neural Data")
hold off

ax(2,2) = subplot(2,4,6:8);

avgs   = squeeze(mean(neural_errors))';
stdevs = squeeze(std(neural_errors))';

b = bar(avgs); hold on
xpos = {b.XEndPoints};

errorbar(xpos{1},avgs(:,1),stdevs(:,1),'k','Linewidth',3,'LineStyle','none')
errorbar(xpos{2},avgs(:,2),stdevs(:,2),'k','Linewidth',3,'LineStyle','none')

xlabel(ax(2,2),"Neural Time Bin for Decoding (ms)")

hold off

mnn  = mean([xpos{1};xpos{2}]);
lbls = num2str(time_bins_sample');

set(ax(:,2),"XTick",mnn,"XTickLabels",lbls)

ylim(ax(1,:),[-.5 1])
ylim(ax(2,:),[0 .7])

set(ax,"Box","off")

if 0,delete(findall(0,"type","errorbar")),end

end

function plot_PAN_data(ax,pan_corrs,pan_errors,time_bins_sample)

ax(1,2) = subplot(2,4,2:4);

avgs   = squeeze(mean(pan_corrs))';
stdevs = squeeze(std(pan_corrs))';

b = bar(avgs); hold on
xpos = {b.XEndPoints};

errorbar(xpos{1},avgs(:,1),stdevs(:,1),'k','Linewidth',3,'LineStyle','none')
errorbar(xpos{2},avgs(:,2),stdevs(:,2),'k','Linewidth',3,'LineStyle','none')

title(ax(1,2),"PAN Data")
hold off

ax(2,2) = subplot(2,4,6:8);

avgs   = squeeze(mean(pan_errors))';
stdevs = squeeze(std(pan_errors))';

b = bar(avgs); hold on
xpos = {b.XEndPoints};

errorbar(xpos{1},avgs(:,1),stdevs(:,1),'k','Linewidth',3,'LineStyle','none')
errorbar(xpos{2},avgs(:,2),stdevs(:,2),'k','Linewidth',3,'LineStyle','none')

xlabel(ax(2,2),"PAN Time Bin for Decoding (ms)")

hold off

mnn  = mean([xpos{1};xpos{2}]);
lbls = num2str(time_bins_sample');

set(ax(:,2),"XTick",mnn,"XTickLabels",lbls)

ylim(ax(1,:),[-.5 1])
ylim(ax(2,:),[0 .7])

set(ax,"Box","off")

if 0,delete(findall(0,"type","errorbar")),end


end

function plot_oracle_data_time(oracle_corrs,oracle_errors,time_bins_sample)

subplot(211)

avgs   = squeeze(mean(oracle_corrs))';
stdevs = squeeze(std(oracle_corrs))';

b = bar(avgs); hold on
xpos = {b.XEndPoints};

errorbar(xpos{1},avgs(:,1),stdevs(:,1),'k','Linewidth',3,'LineStyle','none')
errorbar(xpos{2},avgs(:,2),stdevs(:,2),'k','Linewidth',3,'LineStyle','none')

title("Oracle Decoding Data")
ylabel("Performance")
hold off

subplot(212)

avgs   = squeeze(mean(oracle_errors))';
stdevs = squeeze(std(oracle_errors))';

b = bar(avgs); hold on
xpos = {b.XEndPoints};

errorbar(xpos{1},avgs(:,1),stdevs(:,1),'k','Linewidth',3,'LineStyle','none')
errorbar(xpos{2},avgs(:,2),stdevs(:,2),'k','Linewidth',3,'LineStyle','none')

xlabel("Ground Truth ICs @ Time Bin t, used for Decoding (ms)")
ylabel("Error")
hold off

mnn  = mean([xpos{1};xpos{2}]);
lbls = num2str(time_bins_sample');

ax = findall(gcf,"type","axes");

set(ax,"XTick",mnn,"XTickLabels",lbls)

% ylim(ax(1,:),[-.5 1])
% ylim(ax(2,:),[0 .7])

set(ax,"Box","off")

if 0,delete(findall(ax,"type","errorbar")),end


end











