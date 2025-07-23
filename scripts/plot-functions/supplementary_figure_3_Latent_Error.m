%% Load prnns from run and black-box RNN data

rootdir    = '/Users/danielcalbick/science/yildirim_lab/pRNN_projects/pong_project/';

output_dir = fullfile(rootdir,"hpc-outputs/");
addpath(genpath('../../'))


prnndirs = [
    "../../hpc-outputs/programmed-networks/28-Feb-2025_300neurons_79a1c769-a392-4e44-bb5d-808ad8499eb0" % 300 neurons
    "../../hpc-outputs/programmed-networks/28-Feb-2025_500neurons_b38419a8-f536-4cf5-923b-26be74f13064" % 800 neurons
    "../../hpc-outputs/programmed-networks/28-Feb-2025_1000neurons_44a04e02-6fc5-4389-9109-584de368715b" % 1000 neurons
];

tpts   = [1 5 10 20 500];
nruns   = numel(prnndirs);
files   = cell(nruns,1);
nfiles  = zeros(nruns,1);
ntpts   = numel(tpts);
outputs = cell(nruns);

ntrials = 79;

for i = 1:nruns
    
    files{i}  = dir(fullfile(prnndirs(i),'rnn-states/*.mat'));
    nfiles(i) = numel(files{i});

    trajectories = struct();
    for k = 1:ntpts
        trajectories.(['traj_x_' num2str(tpts(k))]) = nan(tpts(k),nfiles(i),ntrials);
        trajectories.(['traj_y_' num2str(tpts(k))]) = nan(tpts(k),nfiles(i),ntrials);    
    end

    outputs{i} = trajectories;

end


for i = 1:nruns
fprintf('%d/%d\n',i,nruns)

    f = files{i};

    for j = 1:nfiles(i)
    fprintf('\t%d/%d\n',j,nfiles(i))

        states = load(fullfile(f(j).folder,f(j).name));
        prnn = states.prnn_validation;
        prnn = prnn.run_network();

        boardState = cellfun(@(m)...
            prnn.W(3:4,:)*m(:,1:max(tpts)),prnn.network_states_raw,...
            uniformoutput=0);

        boardState = cat(3,boardState{:});
        for k = 1:ntpts     
            outputs{i}.(['traj_x_' num2str(tpts(k))])(:, j,:) = boardState(1, 1:tpts(k),:);
            outputs{i}.(['traj_y_' num2str(tpts(k))])(:, j,:) = boardState(2, 1:tpts(k),:);
        end

    end

end



%{

Temporal roll out

do RMSE for IC
do RMSE for 1  time steps
            5  time steps
            10 time steps

%}

%% 

trial = 12;
conversion = prnn.board_params.rishi_conversion^(-1);
xdata = prnn.data.sim_coordinates.x{trial}(1:50:500);
ydata = prnn.data.sim_coordinates.y{trial}(1:50:500);


for i= flip(1:3)
x = outputs{i}.traj_x_500(:,1,trial);
y = outputs{i}.traj_y_500(:,1,trial);

plot(x*conversion,y*conversion),hold on

end

scatter(xdata,ydata,'o','cyan')
v = [10 10 ; -10 10 ; -10 -10 ; 10 -10 ; nan nan];
f = 1:5;

patch('Faces',f,'Vertices',v,"LineWidth",2)
set(gca,"Visible",0);

legend(["300 Units" "500 Units" "1000 Units" "Ball Traj"])

hold off
axis equal

%%

conversion = prnn.board_params.rishi_conversion;
% get the analytic position of the ball with IC (x,y,x_dot, y_dot) across
% all 79 trials during the first 500 ms of trial
ballx = cellfun(@(x) x(1:500),prnn.data.sim_coordinates.x,UniformOutput=0);
ballx = cat(2,ballx{:})*conversion;
bally = cellfun(@(y) y(1:500),prnn.data.sim_coordinates.y,UniformOutput=0);
bally = cat(2,bally{:})*conversion;

% dim ballx/bally is [ntpts , ntrials]

for i = 1:nruns

    % get the simulated ball position for this model family over the same
    % time period, dim [ntpts , nmodels, ntrials]
    simx = outputs{i}.traj_x_500;
    simy = outputs{i}.traj_y_500;

    %---- Put RMSE computation here ----%
end

%---- Put Plotting Code Here ----%
% Use st. dev across intra-family models and trials, as patch(...)
% error bars 

%%

maxT = 7;
t = 1:maxT;
conversion = prnn.board_params.rishi_conversion;

% get the analytic position of the ball with IC (x,y,x_dot, y_dot) across
% all 79 trials during the first 500 ms of trial
ballx = cellfun(@(x) x(t),prnn.data.sim_coordinates.x,UniformOutput=0);
ballx = cat(2,ballx{:})*conversion;
bally = cellfun(@(y) y(t),prnn.data.sim_coordinates.y,UniformOutput=0);
bally = cat(2,bally{:})*conversion;
% Setup figure
figure('Position', [100, 100, 800, 600]);
colors = colororder;

ntpts = numel(t);

analyPos = [sqrt(reshape(ballx,ntpts,[]).^2+reshape(bally,ntpts,[]).^2)];


% Single loop through model families
for i = flip(1:nruns)

    % Get simulated trajectories
    % [ntpts, nmodels, ntrials]
    simx = permute(outputs{i}.traj_x_500,[1 3 2]);  
    simy = permute(outputs{i}.traj_y_500,[1 3 2]);   
   
    simx = simx(1:maxT,:,:); 
    simy = simy(1:maxT,:,:);

    % Get actual ball trajectory
    bxdata = repmat(ballx,1,1,size(simx,3));
    bydata = repmat(bally,1,1,size(simx,3));

    % Compute the root squared error of x and y positions
    distError  = sqrt((bxdata-simx).^2 + (bydata-simy).^2);

    % Take the mean and standard deviation
    % of that error across trials and models
    RMSE  =  mean(distError,[2 3]);
    stdev =   std(distError,[],[2 3]);
    
    % Plot mean trajectory error
    plot(t, RMSE, 'Color', colors(i,:), 'LineWidth', 2); hold on
    
    % Plot std as patch 
    patch([t, fliplr(t)], ...
          [RMSE'+stdev', flip(RMSE'-stdev')], ...
          colors(i,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none');

end





%%

j = 3; j = 1;

files = dir(fullfile(prnndirs(j),'rnn-states/*.mat'));
nfiles = numel(files);

states = load(fullfile(files(j).folder,files(j).name));
prnn = states.prnn_validation;
prnn = prnn.run_network();

outputs = prnn.W*prnn.network_states_raw{1};


ax = plot_setup(prnn);
plot(ax,outputs(3,1:3000),outputs(4,1:3000))

%%
function ax = plot_setup(self)
            figure('Position',[570 328 892 748],'Name',"pRNN Pong");
            ax = gca;hold(ax,'on')

            xo    = self.board_params.buffer;
            yw    = self.board_params.y_width;
            xw    = self.board_params.x_width;
            ud.pxy   = self.board_params.paddle_fix_value;

            ud.paddle_var_index = self.board_params.paddle_var_index;
            ud.paddle_var_lims  = self.board_params.paddle_var_limits;

            % ps  = self.board_params.sensativity;
            
            wallx = self.board_params.plot_wall_x;
            wally = self.board_params.plot_wall_y;
            
            
            switch self.board_params.paddle_pos
                case {'left' 'right'}
                    pfuncx = @(b,pxy) b(13,:);
                    pfuncy = @(b,pxy) repmat(pxy,size(b,2),1);
                case {'top' 'bottom'}
                    pfuncx = @(b,pxy) repmat(pxy,size(b,2),1); 
                    pfuncy = @(b,pxy) b(13,:);        
            end

            ud.pfuncx = pfuncx;
            ud.pfuncy = pfuncy;
             
            ballsz = 30;  

            % Plot Wall
            plot(ax, wallx, wally,'k-', 'linewidth',4,...
                 'clipping',0);

            % Ball        
            scatter(ax, nan, nan, ballsz, 'filled','MarkerFaceAlpha',.3,'Tag','ball'); 
            scatter(ax, nan, nan, ballsz, 'k', 'filled','Tag','ball_t0',...
                'MarkerFaceAlpha',0.3,'MarkerEdgeColor','r','LineWidth',2);
                     
            % Paddle
            scatter(ax, nan ,nan , ballsz, "square",'filled','Tag','paddle');
            
            set(ax, 'UserData',ud, 'Visible' , false,...
                'XLim' , [-1 1]*(yw+xo) , 'YLim' , [-1 1]*(xw+xo)  )           
            

            ax.UserData = ud;
end