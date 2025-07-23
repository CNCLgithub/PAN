%% Explore 180º

p = 3;  % if bigger makes a torus knot
q = 5; % if bigger wraps only around fiber

R = -1;
r = 1;

nt = 3600;
t  = linspace(0,2*pi,nt)';

x = (R + r*cos(q*t)) .* cos(p*t);
y = (R + r*cos(q*t)) .* sin(p*t);
z = r * sin(q*t);



% hues = linspace(0,300,nt)'/360;
divt = floor(nt/5);
ext  = .2;

phs = .5;

ncolor     = floor(ext*nt);
nwhite     = nt - 2*ncolor;
centerIdx2 = ceil(nt/2);


huevec = [ 0 .5];
nh = numel(huevec);

hue1 = ones(ncolor,1)*huevec(1);
hue2 = ones(ncolor,1)*huevec(2);

satsc = sin(linspace(0,pi,ncolor)');

vals = ones(floor(ext*nt),1)*.8;

clr1 = hsvf(hue1,satsc,vals);
clr2 = hsvf(hue2,satsc,vals);

hues = [ones(floor(nt/2),1)*huevec(1) ; ones(ceil(nt/2),1)*huevec(2)] ;
hues = circshift(hues,-round(nt/4));

sats = zeros(nt,1);
sats(1:ncolor) = satsc;
sats(centerIdx2:centerIdx2+ncolor-1) = satsc;
sats = circshift(sats,-round(ncolor/2));

vals = ones(nt,1)*.8;

clrs = hsvf(hues,sats,vals);

p0 = findall(0,'-depth',3,'tag','torus');
if ~isempty(p0), delete(p0), end


faces = 1:nt+1;
verts = [[x y z] ; nan(1,3)];
clrs(end+1,:) = [0 0 0];

patch('Faces',faces,'Vertices',verts,...
    'FaceColor','none','FaceVertexCData',clrs, ...
    'EdgeColor','flat','LineWidth',5, ...
    'Marker','.','MarkerFaceColor','flat', ...
    'MarkerSize',50, ...
    'FaceLighting','gouraud', 'tag','torus');

hold on
% Create parameter arrays
t = linspace(0, 2*pi, nt);
[T, P] = meshgrid(t, t);  % Create 2D parameter grids

% Calculate torus coordinates
xcor = (R + r*0.8*cos(q*P)) .* cos(p*T);
ycor = (R + r*0.8*cos(q*P)) .* sin(p*T);
zcor = r*0.8 * sin(q*P);


% Plot the surface
ap = abs(p);aq = abs(q);
alpha = min(ap,aq)/max(ap,aq);
alpha = min(.2,alpha);
alpha = max(.1,alpha);

surf(xcor, ycor, zcor, ...
    'EdgeColor','none','FaceColor','k',...
    'FaceAlpha',alpha, 'tag','torus')

set(gca,'Visible','off')

% view([-28.2746   21.2939])


%%


p = 4; % if bigger makes a torus knot
q = 3; % if bigger wraps only around fiber



R = 2;
r = 1;

nt = 360;
t = linspace(-pi,pi,nt)';

x = (R + r*sin(q*t)) .* cos(p*t);
y = (R + r*sin(q*t)) .* sin(p*t);
z = r * cos(q*t);

huevec = [0 60 120 180 240 300]./360;
% huevec = [0  180 ]./360;
%{

[  0   60   120  180  240 300 360]
[  r    y    g    c    b   m   r

[-360 -300 -240 -180 -120 -60  0]
[  r    g    y    c    m   b   r]


%}
nh = numel(huevec);


divt = floor(nt/nh);
ext  = .8;

ncolor     = floor(ext*divt);
nwhite     = nt - 2*ncolor;
centerIdx2 = ceil(nt/2);

huemat = ones(divt,1)*huevec;
hues   = circshift(huemat(:),-round(ncolor/2)*0);


win   = gausswin(divt,1);
satsc = repmat(sin(linspace(0,pi,divt)'),1,nh).*win;

sats   = circshift(satsc(:),-round(ncolor/2)*0);

% sats(1:ncolor) = satsc;
% sats(centerIdx2:centerIdx2+ncolor-1) = satsc;
% sats = circshift(sats,-round(ncolor/2));

vals = ones(nt,1)*.8;

clrs    = hsvf(hues,sats,vals);
setView = true;

p0 = findall(0,'-depth',3,'tag','torus');
if ~isempty(p0), delete(p0), setView=false; end


faces = 1:nt+1;
verts = [[x y z] ; nan(1,3)];
clrs(end+1,:) = [0 0 0];

LW = 15;
patch('Faces',faces,'Vertices',verts,...
    'FaceColor','none','FaceVertexCData',clrs, ...
    'EdgeColor','flat','LineWidth',LW, ...
    'FaceLighting','gouraud', 'tag','torus');
    % 'Marker','.','MarkerFaceColor','flat', ...
    % 'MarkerSize',50, ...

hold on
% Create parameter arrays
t = linspace(0, 2*pi, nt);
[T, P] = meshgrid(t, t);  % Create 2D parameter grids

% Calculate torus coordinates
xcor = (R + r*0.9*sin(q*P)) .* cos(p*T);
ycor = (R + r*0.9*sin(q*P)) .* sin(p*T);
zcor = r*0.9 * cos(q*P);

if 0
qq = 2;
pp = 1;
fibert = interp1(1:5,[0 pi/2  2*pi/3  pi/2  2*pi],1:nt )';
ringt = linspace(0,2*pi,30)';
trajx = (-2 + 1.5*r*sin(qq*ringt)) .* cos(pp*ringt); 
trajy = (2 + 1.5*r*sin(qq*ringt)) .* sin(pp*ringt); 
trajz = 1.2*r * cos(qq*ringt);

nnt = numel(ringt);
hues   = ones(nnt,1)*(60/360);
sats   = ones(nnt,1);
vals   = linspace(1,.5,nnt)';
clrs    = hsvf(hues,sats,vals);

faces = 1:nnt+1;
verts = [[trajx trajy trajz] ; nan(1,3)];
clrs(end+1,:) = [0 0 0];

patch('Faces',faces,'Vertices',verts,...
    'FaceColor','none','FaceVertexCData',clrs, ...
    'EdgeColor','flat','LineWidth',3, ...
    'Marker','o','MarkerSize',20,'MarkerFaceColor','flat',...
    'FaceLighting','gouraud', 'tag','torus');
end


% Plot the surface
ap = abs(p);aq = abs(q);
alpha = min(ap,aq)/max(ap,aq);
alpha = min(.2,alpha);
alpha = max(.1,alpha);

alpha = .4;

surf(xcor, ycor, zcor, ...
    'EdgeColor','none','FaceColor','k',...
    'FaceAlpha',alpha, 'tag','torus')


I3 = [eye(3) ; - eye(3)];

ux = [zeros(6,1) I3(:,1)];
uy = [zeros(6,1) I3(:,2)];
uz = [zeros(6,1) I3(:,3)];

plotAxis = false;


nhues = numel(huevec);
if isempty(findall(gca,'tag','uax'))
    if plotAxis
        line(ux',uy',uz','LineWidth',4,'tag','uax')
        set(gca,'ColorOrder',hsvf(huevec',ones(nhues,1),ones(nhues,1)))
    else, delete(findall(gca,'tag','uax'))
    end
end


set(gca,'Visible','off')
if setView, view(3),end
% view([-28.2746   21.2939])


%%


p = 4; % if bigger makes a torus knot
q = 3; % if bigger wraps only around fiber



R = 2;
r = 1;

nt = 360;
t = linspace(-pi,pi,nt)';


% huevec = [0 60 120 180 240 300]./360;
huevec = [0  180 ]./360;
%{

[  0   60   120  180  240 300 360]
[  r    y    g    c    b   m   r

[-360 -300 -240 -180 -120 -60  0]
[  r    g    y    c    m   b   r]


%}
nh = numel(huevec);


divt = floor(nt/nh);
ext  = .8;

ncolor     = floor(ext*divt);
nwhite     = nt - 2*ncolor;
centerIdx2 = ceil(nt/2);

huemat = ones(divt,1)*huevec;
hues   = circshift(huemat(:),-round(ncolor/2)*0);

win   = gausswin(divt,3);
satsc = repmat(sin(linspace(0,pi,divt)'),1,nh).*win;
sats  = circshift(satsc(:),-round(ncolor/2)*0);

vals = ones(nt,1)*.8;

clrs    = hsvf(hues,sats,vals);
setView = true;

p0 = findall(0,'-depth',3,'tag','torus');
if ~isempty(p0), delete(p0), setView=false; end

% Calculate torus knot coordinates
x = (R + r*sin(q*t)) .* cos(p*t);
y = (R + r*sin(q*t)) .* sin(p*t);
z = r * cos(q*t);


faces = 1:nt+1;
verts = [[x y z] ; nan(1,3)];
clrs(end+1,:) = [0 0 0];

LW = 10;
pknot=patch('Faces',faces,'Vertices',verts,...
    'FaceColor','none','FaceVertexCData',clrs, ...
    'EdgeColor','flat','LineWidth',LW, ...
    'FaceLighting','gouraud', 'tag','torus');
    % 'Marker','.','MarkerFaceColor','flat', ...
    % 'MarkerSize',50, ...

hold on
% Create parameter arrays
t2 = linspace(0, 2*pi, nt);
[T, P] = meshgrid(t2, t2);  % Create 2D parameter grids

% Calculate torus surface coordinates
xcor = (R + r*0.8*sin(q*P)) .* cos(p*T);
ycor = (R + r*0.8*sin(q*P)) .* sin(p*T);
zcor = r*0.8 * cos(q*P);


% Plot the surface
ap = abs(p);aq = abs(q);
alpha = min(ap,aq)/max(ap,aq);
alpha = min(.2,alpha);
alpha = max(.1,alpha);

ptorus = surf(xcor, ycor, zcor, ...
    'EdgeColor','none','FaceColor','k',...
    'FaceAlpha',alpha, 'tag','torus');


I3 = [eye(3) ; - eye(3)];
ux = [zeros(6,1) I3(:,1)];
uy = [zeros(6,1) I3(:,2)];
uz = [zeros(6,1) I3(:,3)];

plotAxis = false;
nhues = numel(huevec);
if isempty(findall(gca,'tag','uax'))
    if plotAxis
        line(ux',uy',uz','LineWidth',4,'tag','uax')
        set(gca,'ColorOrder',hsvf(huevec',ones(nhues,1),ones(nhues,1)))
    else, delete(findall(gca,'tag','uax'))
    end
end


set(gca,'Visible','off')
if setView, view(3),end
% view([-28.2746   21.2939])

cycles = 4;
timesteps = 1000;
Tdyn = linspace(-2*pi*cycles,2*pi*cycles,timesteps);

dynamicR = cos(Tdyn);

pause(.5)

for i = 1:numel(Tdyn)


    xcort = (dynamicR(i) + r*sin(q*t)) .* cos(p*t);
    ycort = (dynamicR(i) + r*sin(q*t)) .* sin(p*t);

    verts = [[xcort,ycort,z]; nan(1,3)];

    % Calculate torus surface coordinate
    xcortS = (dynamicR(i) + r*0.8*sin(q*P)) .* cos(p*T);
    ycortS = (dynamicR(i) + r*0.8*sin(q*P)) .* sin(p*T);

    set(pknot,"Vertices",verts)
    set(ptorus,"Xdata", xcortS,"ydata",ycortS)

    drawnow



end