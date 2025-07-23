%% Setup Environment

[rootdir,systDirString] = setup_PAN_environment();

panFile = "hpc-outputs/programmed-networks/" + ...
"28-Feb-2025_1000neurons_44a04e02-6fc5-4389-9109-584de368715b/" +...
"rnn-states/prnn-states-id_49_05630dcb-2eec-4830-b764-e059d6a56bc3.mat";

filepath = fullfile(rootdir,panFile);
filepath = replace(filepath,"/",systDirString);

PANdata = load(filepath);
prnn    = PANdata.prnn_validation.run_network();

%% Create Animation

% If set to true, this will create and save the individual .png files to
% the animation folder in rootdir/figures. If you want to make a .gif from
% these .pngs there are online tools as well as command line tools like
% ffmpeg that can do this.
%

printImage = false;

if printImage
    savefldr = fullfile(rootdir,"figure","animation");
    if ~exist(savefldr,"dir"),mkdir(savefldr),end
end

% Choose pong condition to animate from 1:79
trial = 6;

% Set animation temporal resolution
dt = 50;

% PNG resolution 
% (100 = 100% match figure resolution, 300 = upsampled to 300%)
pngres = 100;

% setup figure handle
f = findall(0,'-depth',1,'name','Pong Animation');
if isempty(f)
    figure(Position=[92 29 1268 1149],Name="Pong Animation")
else,delete(f.Children)
end
ax = gobjects(3,1);


tdata = 250:2500;

statemat  = prnn.network_states_raw{trial}(:,tdata);
statevars = prnn.W*statemat;

xdata = statevars( 3,:);
ydata = statevars( 4,:);
sry1  = statevars( 7,:);
sry2  = statevars( 8,:);
colt  = statevars(11,:);


%--------- setup nullcline phase portrait
xvec = linspace(-.2,.2,1000);
fsr1 = @(xcol,sr2) (xvec/5) - (2020*xvec.^3) + 100*(10*xcol^2 - 1/5)*(sr2 - 1/10) - 2;
fsr2 = @(xcol,sr1) (xvec/5) - (2020*xvec.^3) + 100*(10*xcol^2 - 1/5)*(sr1 - 1/10) - 2;
fcol = @(ypos) 1000*xvec*(ypos - 9/100) - 1000*xvec.^3;

colors = colororder;
xvarc = hsvf([180 0 60]'/360,[1 1 1]',[1 1 1]');
ax(1) = subplot(2,2,[1 3]); 

psr1  = plot(ax(1),xvec,fsr1(colt(1),sry2(1)),'Color',colors(1,:));hold(ax(1),"on")
psr2  = plot(ax(1),xvec,fsr2(0,sry1(1)),'Color',colors(2,:));
pcol  = plot(ax(1),xvec,fcol(ydata(1)),'Color',colors(3,:));

ballsize = 100;
vsr1  = scatter(ax(1),sry1(1),0,ballsize,'CData',xvarc(1,:), ...
    "MarkerEdgeColor",'k');
vsr2  = scatter(ax(1),sry2(1),0,ballsize,'CData',xvarc(2,:), ...
    "MarkerEdgeColor",'k');
vcol  = scatter(ax(1),colt(1),0,ballsize,'CData',xvarc(3,:), ...
    "MarkerEdgeColor",'k');

dpx = -.04;

velpntc = text(ax(1),sry1(1)+dpx,-.3,"$\pm v=z_{7}$", ...
    "Interpreter","latex","Color",hsvf(.5,1,.8), ...
    'FontSize',24,"HorizontalAlignment","center");


xlim(ax(1),[-1 1]*.15)
ylim(ax(1),[-1 1]*2.7)
xlab = text(ax(1),.15, .25 ,"$z_i$", ...
    "Interpreter","latex","Color",'k', ...
    'FontSize',24,"HorizontalAlignment","center");

ylab = text(ax(1),.02, 2.5,"$\dot{z}_i$", ...
    "Interpreter","latex","Color",'k', ...
    'FontSize',24,"HorizontalAlignment","center");

set(ax(1),"Box","off","XAxisLocation","origin","YAxisLocation","origin")
xticks([]),yticks([])

strs = [
    "$\dot{z}_7$"
    "$\dot{z}_8$"
    "$\dot{z}_{11}$"
    "$z_7(t)$"
    "$z_8(t)$"
    "$z_{11}(t)$"
];

title(ax(1),"Phase Portraits of the"+newline+ ...
    "Closed-loop Collision Ciruit Manifolds")

legend([psr1 psr2 pcol vsr1 vsr2 vcol],strs, ...
    "Location","southwest","Interpreter","latex", ...
    "Box",0,"NumColumns",2)

%--------- setup state-variable observables

wallx = prnn.board_params.plot_wall_x;
wally = prnn.board_params.plot_wall_y;

pause(.1)
ax(2) = subplot(2,2,2);
plot(ax(2),wallx, wally,'k'),hold(ax(2),"on")

tr   = plot(ax(2),nan,nan,'magenta');
ball = scatter(ax(2),xdata(1),ydata(1),80,'k');

set(ax(2),"visible","off")
xlim(ax(2),[-1 1]*.11)
ylim(ax(2),[-1 1]*.11)
axis(ax(2),"equal");

title(ax(2),"Simulatation State-Variable Read-Out")

% --------- setup latent-variable point-cloud
nlatents = size(statemat,1);
unitsize = 80;

% Generate uniform distribution within unit disc
theta = 2 * pi * rand(nlatents, 1);
r_base = sqrt(rand(nlatents, 1));

% Add Gaussian noise for "slop"
noise_std = 0.05;  % Standard deviation of noise (adjust for more/less)
r = r_base + noise_std * randn(nlatents, 1);

% Convert to Cartesian
vecx = r .* cos(theta);vecy = r .* sin(theta);

normstate = zscore(statemat,[],2);
ax(3) = subplot(2,2,4);
pcloud = scatter(ax(3),vecx,vecy,unitsize, ...
    "CData",normstate(:,1),"MarkerEdgeColor","k",MarkerEdgeAlpha=.8);

colormap(slanCM("coolwarm"))
set(ax(3),"visible","off")
xlim(ax(3),[-1 1])
ylim(ax(3),[-1 1])
axis(ax(3),"equal");
cbar = colorbar;
set(cbar,"Position",cbar.Position + [.07 0 0 0],...
    "Ticks",[-2 2], "TickLabels",["$-$" "$+$"],...
    "TickLabelInterpreter","latex")
cbartxt = text(ax(3),cbar.Position(1)+.55,cbar.Position(2)-.12, ...
    "Hidden Unit Activation","HorizontalAlignment","center", ...
    "Rotation",270,"FontSize",15);

clim([-1 1]*2)

ttl = title(ax(3),"Hidden State Time Evolution");
ttl.Position = ttl.Position + [0 .3 0];


tsample  = 2:dt:numel(xdata);
nsample  = numel(tsample);
filename = @(n) fullfile(savefldr, ...
    sprintf("pong-frame-%0"+numel(num2str(nsample))+"d.png",n));

ax(2).XAxis.Visible = 0;
ax(2).YAxis.Visible = 0;
ax(3).XAxis.Visible = 0;
ax(3).YAxis.Visible = 0;

set(ax(2:3),"Box","off","Visible",1)


for i = 1:numel(tsample)

    t = tsample(i);

    set(psr1,"YData",fsr1(colt(t),sry2(t)));
    set(psr2,"YData",fsr2(0,sry1(t)));
    set(pcol,"YData",fcol(ydata(t)));

    set(vsr1,"XData",sry1(t))
    set(vsr2,"XData",sry2(t))
    set(vcol,"xData",colt(t))
    velpntc.Position(1) = sry1(t)+dpx;

    set(pcloud,"CData",normstate(:,t));

    set(ball,"XData",xdata(t),"YData",ydata(t));
    set(tr,"XData",xdata(2:t),"YData",ydata(2:t));

    if printImage
        exportgraphics(gcf, filename(i),'Resolution',pngres,"Padding","figure");
    else, pause(.1) 
    end



end





