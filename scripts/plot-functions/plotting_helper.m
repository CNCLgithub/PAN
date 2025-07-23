%% Trial Plotting Functions
function varargout = plotting_helper(plot_type, rnndata, ax, bp)

axtrig = ~exist("ax","var") || isempty(ax);

switch plot_type

    case 'plot_game_trial'

        if axtrig,ax = figure_setup(bp);end
        plot_game_trial(rnndata, ax);

    case 'plot_trial_set'

        if axtrig, ax = figure_setup(bp); end
        plot_trial_set(rnndata, ax);

    case 'plot_mapping_data'
        plot_mapping_data(rnndata);
        ax = gca;

end

if nargout, varargout = {ax};end



end

function ax = figure_setup(board_params)
        
    figure('Position',[570 328 892 748],'Name',"pRNN Pong");
    ax = gca;hold(ax,'on')

    xo       = board_params.buffer;
    yw       = board_params.y_width;
    xw       = board_params.x_width;
    ud.pxy   = board_params.paddle_fix_value;

    ud.paddle_var_index = board_params.paddle_var_index;
    ud.paddle_var_lims  = board_params.paddle_var_limits;

    % ps  = self.board_params.sensativity;
    
    wallx = board_params.plot_wall_x;
    wally = board_params.plot_wall_y;
    
    switch board_params.paddle_pos
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
    hold(ax,'off')

    ax.UserData = ud;


end

function plot_game_trial(data, ax)

    ud     = ax.UserData; 

    b      = data;         
    fnidx  = size(b,2);    

    x      = b(3,1:fnidx);
    y      = b(4,1:fnidx);
    py     = ud.pfuncx(b,ud.pxy);
    px     = ud.pfuncy(b,ud.pxy);
    nx     = numel(x);

    spacer = (randn(nx,1)*0.001);
    switch ud.paddle_var_index
        case 3,py = py + spacer;
        case 4,px = px + spacer;
    end
         
    pdlc   = parula(numel(px));
                           
    ball   = findall(ax,'-depth',1,'Tag','ball');
    ball0  = findall(ax,'-depth',1,'Tag','ball_t0');
    paddle = findall(ax,'-depth',1,'Tag','paddle');

    set(ball  , 'xdata',x,'ydata',y,'cdata',pdlc)
    set(ball0 , 'xdata',x(1),'ydata',y(1),'cdata',pdlc(1,:))
    set(paddle, 'xdata',px,'ydata',py,'cdata',pdlc)

end

function plot_trial_set(data , ax)

    ud = ax.UserData; 

    outputs = data;
    ntrials = numel(outputs);

    ball   = findall(ax,'-depth',1,'Tag','ball');
    ball0  = findall(ax,'-depth',1,'Tag','ball_t0');

    ballc  = turbo(ntrials);

    for i = 1:ntrials

        b      = data{i};         
        fnidx  = size(b,2);    

        x      = b(3,1:fnidx);
        y      = b(4,1:fnidx);
        nx     = numel(x);                      
                               
        grey   = [linspace(255, 100, nx)', linspace(255, 100, nx)', linspace(255, 10, nx)'];
        clr    = (ballc(i,:).*grey)/255;

        bcopy  = copyobj(ball ,ax);
        b0copy = copyobj(ball0,ax);
        
        set(b0copy , 'xdata',x(1),'ydata',y(1),'cdata',clr(1,:))
        set(bcopy , 'xdata',x,'ydata',y,'cdata',clr)
    
    end

    paddle = findall(ax,'-depth',1,'Tag','paddle');
    if strcmp(get(paddle,'type'),"scatter")
        delete(paddle);
        paddle = patch(ax , 'XData' , nan , 'YData' , nan,...
            'FaceColor','k','FaceAlpha',0.2,'Tag','paddle');
    end
    
    fix  = ones(1,4)*ud.pxy + [-1 1 1 -1]*.002;
    vari = [-1 -1 1 1]*ud.pxy;
    switch ud.paddle_var_index
        case 3, px = vari; py =  fix;
        case 4, px =  fix; py = vari;          
    end

    set(paddle, 'xdata',px,'ydata',py)


end

%% Correlation Plotting

function plot_mapping_data(map)
%%
tdec  = map.t_decode;
tpred = map.t_validation;

tlts = {'Training Error','Validation Error',...
     'Prediction Correlation' , 'Sum Metric, (-2||MSE||-1) + Correlation'};

figure('Position',[214 30 1540 1166]);
ax(1) = subplot(221);
ax(2) = subplot(222);
ax(3) = subplot(223);
ax(4) = subplot(224);

terr = map.avg_train_error;
verr = map.avg_valid_error;
pcor = map.avg_valid_correlation;
psum = map.avg_valid_cor_mse_sum;

data{1} = imgaussfilt(terr,50);
data{2} = imgaussfilt(verr,50);
data{3} = imgaussfilt(pcor,50);
data{4} = imgaussfilt(psum,50);

best_idx = map.best_map_index;

xvec = {tdec};
xvec(2:4) = repmat({tpred},1,3);
   
nplts = numel(data);

for i = 1:nplts

    imagesc(xvec{i},tdec,data{i},'Parent',ax(i) )
    title(ax(i),tlts{i})
    colorbar(ax(i))

    if i == 1, ylabel(ax(i),'Decoding Timepoint Validation'), end
    if nplts > 1 && i == 2, xlabel(ax(i),'Decoding Timepoint (Map)')
    elseif nplts == 1, xlabel(ax(i),'Decoding Timepoint (Map)'),end
    
end

set(ax,'YDir','normal')
p = parula;
cmap_err  = tintedParulaGradient;
cmap_cor  = p;
cmap_comb = cmap_err(1:end/2,:);

set(ax(1:2), 'CLim' , [-1 1]*.04,'colormap',cmap_err)
set(ax(3), 'CLim' , [-1 1]*.8,'colormap'  ,cmap_cor)
set(ax(4), 'CLim' , [-.05 1.8],'colormap' ,cmap_comb)
sgtitle('Smoothed, Normalized, and Averaged-across-trials')
hold(ax,"on")
avg   = mean(psum,2); 
avg   = (avg-min(avg)); avg = avg/max(avg);
xst   = ax(4).XLim(2)+1;
dfx   = abs(diff(ax(4).XLim)/10);
x     = linspace(xst,xst+dfx,numel(avg));
xq    = linspace(xst,xst+dfx,dfx);
avg   = interp1(x,avg,xq);
avg   = (avg*dfx)+xst;
y     = linspace(1,numel(x),numel(avg));
plot(ax(4) , avg,y)
set(ax(4), 'XLim' , [1 avg(end)]);
pf = @(a) plot(a , a.XLim , [1 1]*best_idx,'g--');
cellfun(@(a) pf(a) , num2cell(ax))
text(ax(4) , mean(ax(4).XLim)-200 , best_idx-200 , ...
    ['Best Decoding' newline 'Time < 800ms'],...
    'Color','g','FontSize',20,'FontWeight','bold','HorizontalAlignment','center')
hold(ax,"off")
set(ax,'Box','off')


end

function parula_grad = tintedParulaGradient(numColors , tint_c)

            if ~exist("numColors","var"), numColors = 256; end
            if ~exist("tint_c","var"), tint_c = [1, 0.8, 0.8]; end
            
            nc = floor(numColors / 2);
            
            % Create the base parula colormap
            parula_half = [parula(nc) ; flipud(parula(nc))];
            
            parula_top    = parula(nc);
            parula_bottom = flipud(parula(nc));
            
            % Define the tints for the top and bottom halves
             % Light red tint for the bottom half
            
            % Apply the gradient tint to the bottom half
            bottom_half = bsxfun(@plus, parula_bottom, bsxfun(@times, tint_c - parula_bottom, linspace(0, 1, nc)'));
            
            % Combine the two halves
            parula_grad = [parula_top ; bottom_half];
            
            % Ensure the colormap has the correct number of colors
            if size(parula_grad, 1) < numColors
                % Add an additional color if needed (for odd numColors)
                parula_grad = [parula_grad; parula_half(end, :)];
            end
            
            
            % cimage(custom_cmap)
        
end


