function [valid_starts,xsv,ysv,counts] = get_valid_pong_starts(ntrials,varargin)
%%
% ntrials = 10e3;
[plottrig, bounce_lims, tlim, min_time] = parse_args_in(varargin);

%%
valid_starts = table('Size',[ntrials,4],'VariableTypes',repmat("double",4,1),'VariableNames',["x" "y" "heading" "speed"]);

counts = nan(ntrials,1);

% rishi_params.xrange = [-10 10];
% rishi_params.yrange = [-10 10];
% rishi_params.heading = 'degrees';
% rishi_params.speed_conversion = 15/20;

[xsv,ysv] = deal(cell(ntrials,1));

% bounce1 = 0;

for i = 1:ntrials
    count = 0;
    while 1
        %%
        count = count+1;
        
        [x0,y0, h0,sp0] = get_new_start();

        [pass,coors] = run_rishi_sim( x0, y0, h0,sp0, bounce_lims);
        
        if tlim, pass = pass & (numel(coors.x{1}) >= min_time); end

        if pass, break,end
       
    end

    % bounce1
    xsv{i} = coors.x{1};
    ysv{i} = coors.y{1};
    counts(i) = count;
    valid_starts(i,:) = {x0,y0, h0,sp0};

end

if plottrig
%%
    plotStuff(xsv,ysv)
end

end

%% Helper Functions

function sandbox()
%%

bounce_lims = [0 1];


[x0,y0, h0,sp0] = get_new_start();

[pass,x,y] = run_sim( x0, y0, h0,sp0 , bounce_lims);

plotStuff({x},{y})

end


function [x0,y0, h0,sp0] = get_new_start()

    h0 = ((2*rand) - 1)*89; % -89 - 89 degrees

    x0  = (2*rand-1)*10;
    y0  = (2*rand-1)*9;
    sp0 = (0.25*randi(3))*15;


end

function [pass,x,y] = run_sim(x0,y0, h0, sp0,bounce_lims)
%%
    
    occluder_width = 0.25;

    xrange = [-10 10];
    yrange = [-10 10];

    % Min/Max for Rishi's perameters in ms
    mn = 624.9;
    mx = 1874.7;

    mxtime = ceil(mx*2); % both conditions vis/occ

    width = diff(xrange);
    
    occ_x = ((1 - occluder_width)*width) + xrange(1);

    deg_per_ms = sp0*(1/1000);

    dx0 = cos(h0)*deg_per_ms;
    dy0 = sin(h0)*deg_per_ms;
        
    game_end = xrange(2);

    [x,y] = deal(mxtime,1);

    x(1) = x0;
    y(1) = y0;

    pass = false; count = 2; bounce = 0;

    while x(count-1) <= game_end 
            
            x(count) = x(count-1) + dx0; 
            y(count) = y(count-1) + dy0;
    
            if y(count) <= yrange(1) || y(count) >= yrange(2)
                dy0 = -dy0;
                bounce = bounce+1;
            end
        
            if bounce == bounce_lims(2)+1,return,end

            if x(count) >= xrange(2), break, end

            count = count+1;
    end

    if bounce == bounce_lims(1)-1,return,end

    x(count+1:end) = [];
    y(count+1:end) = [];

    vis  = x < occ_x;
    nvis = sum(vis);
    nocc = sum(~vis);

    if prod([nvis nocc] >= mn) && prod([nvis nocc] <=mx), pass = true; end

end


function plotStuff(xsv,ysv)
%%

figure
ax = axes(gcf);

hold(ax,'on')

ntr   = numel(xsv);
occ_w = 0.25;

c = parula(ntr);

wx = [10 -10  -10 10];
wy = [10 10 -10  -10];

xrange = [-10 10];
width  = diff(xrange);

occ_x = ((1 - occ_w)*width) + xrange(1);

ox = [10 occ_x  occ_x   10];


patch('XData',ox,'YData',wy,'FaceColor',[0.71 0.1 0.14],'FaceAlpha', 0.3)

plot(wx,wy,'k','LineWidth',4)

for i = 1:numel(xsv)
    plot(xsv{i},ysv{i} , Color=c(i,:));
    scatter(xsv{i}(1),ysv{i}(1),100,"r",'filled');
end

axis(ax,'equal')
xlim([-1 1]*11),ylim([-1 1]*11)

end

function [plottrig, bounce_lims, tlim,min_time] = parse_args_in(args)

% Defaults
plottrig    = 0;
bounce_lims = [0 1];
tlim        = 0;
min_time    = nan;
if ~isempty(args)

    if islogical(args{1}),plottrig = args{1}; args{1} = [];end

    args = reshape(args,2,[]);
    for i = 1:size(args,2)
        switch lower(args{1,i})
            case {'plot' , 'plotstuff'}, plottrig = logical(args{2,i});
            case {'bounce' , 'blims' , 'nbounces' , 'nbounce'},bounce_lims = args{2,i};
            case {'tlimit' , 'time' , 'tlim' }, tlim = true; min_time = args{2,i};
        end

    end
    
end

end

