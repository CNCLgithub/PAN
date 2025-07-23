%% Load Rishi's Data

load([rootpath 'data/perle_dataset.mat'],'perle')
load([rootpath 'data/mahler_dataset.mat'],'mahler')

%% Check direct conversions between "degrees" from paper and pixels

% Degree parameters
x = mahler.meta.ball_offset_x;
y = mahler.meta.ball_offset_y;
h = mahler.meta.ball_heading;
s = mahler.meta.ball_speed;

% Pixel Parameters
xx = mahler.meta.x0;
yy = mahler.meta.y0;
hh = mahler.meta.heading0;
ss = mahler.meta.speed0;

% frame rate conversions
frameRate   = 60;
fs          = 1000/frameRate;
dt          = fs/1000; % seconds

% try conversion (working)
x0 = ((x/10)+1)*16; 
y0 = ((y/10)+1)*16;
h0 = deg2rad(h);

s0 = (s*dt)*4;
% The speed conversion is a bit weird but since the max time they are
% setting for visual and occluded regions is 1874.7 ms --> the max trial
% length is 3,749.4, ie a cieling of 3,750 ms. Since there is a limit of
% one bounce, the max distance traveled is from one right corner to the
% same corner on the left --> sqrt(10º^2 + 20º^2)*2 = 14.1421º ≈ 15º; thus,
% the max velocity per ms is 15º/3750ms = 0.004 º/ms; having 3 slower
% speeds [.25 .50 .75]*max_deg = [3.75 7.50 11.25]
%
% Further, to convert it into a NEW TIME SCALE, we need to take into
% account the dt for updating, which is 0.004 º/ms, and since degree here 

ntrials = numel(x0);

fprintf("x conversion error: %f\n"      , norm(x0 - xx))
fprintf("y conversion error: %f\n"      , norm(y0 - yy))
fprintf("heading conversion error: %f\n", norm(h0 - hh))
fprintf("speed conversion error: %f\n"  , norm(s0 - ss))


%%

idx = 18;

x0  = x(idx);
y0  = y(idx);
h0  = deg2rad(h(idx));
sp0 = s(idx);

occ_w = .25;

[pass,xs,ys] = run_sim( x0,y0,h0,sp0,occ_w);

plotStuff({xs},{ys}, occ_w)

%%

occ_w = linspace(0.1,.6,100);

nocc = numel(occ_w);
psv = false(ntrials,nocc);

close all

for j = 1:nocc

    for idx = 1:ntrials
    

        x0  = x(idx);
        y0  = y(idx);
        h0  = deg2rad(h(idx));
        sp0 = s(idx);
        
        psv(idx,j)= run_sim( x0,y0,h0,sp0,occ_w(j));
    
    end

end

imagesc(occ_w,1:ntrials,psv)


%% Functions
function [pass,x,y] = run_sim(x0,y0, h0, sp0,occluder_width)
%%
    
    bounce_lims = [0 1];

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

function plotStuff(xsv,ysv,occ_w)
%%

figure
ax = axes(gcf);

hold(ax,'on')

ntr = numel(xsv);

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

