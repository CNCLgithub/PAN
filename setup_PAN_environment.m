function [rootdir,systDirString] = setup_PAN_environment()
% This function just adds all of the files within the PAN directory to the
% current matlab path so all scripts and files are visible while running
% the scipts in the different modules.
%
% There is a chance that if a filename is named the same thing as something
% you have in your saved path, matlab will throw a warning that there is a
% conflict; if that happens you can triage by typing
%
% >> which "foo"
%     path/to/this/file/foo.m
%
% Then you can temporarily remove that conflict via 
%
% >> rmpath("path/to/this/file/foo.m")
%
% This will not make any perminant changes to your setup; the next time you
% open MATLAB it will load all of your normal saved path directories and files 
%
%

     [rootdir,~,~] = fileparts(matlab.desktop.editor.getActiveFilename);
     
     if ispc,systDirString = '\';
     else, systDirString = '/';end

     addpath(genpath(rootdir))
     figure_setup()


end


function figure_setup()
%{
This sets up my personal style for matlab defaults. If you like them or
want to custamize your own matlab defaults you can create a file named
"startup.m" add it to your path and save it via

>> addpath("path/to/startup.m"); savepath

%}

% If there is a warning that is annoying you can run:
% [a, MSGID] = lastwarn();
% Just after it happens and then add the result of MSGID into this list

annoyingWarnings = {
    'MATLAB:ui:Slider:fixedHeight'
    };
warning("off",annoyingWarnings{:})

% To look at all adjustable default parameters run this in your command
% window: 
%          default_params = get(0,'factory'); open default_params
% 
% properties are set as: 'defaultObjectPoperty' , some availible properties
% to explore are defaultLine , defaultScatter, defaultArrow, defaultBar,...


my_color_order = [
    0.2980    0.5529    0.7098
    0.5765    0.2392    0.1725
    0.7882    0.7098    0.4588
    0.5569    0.4078    0.7294
    0.0824    0.4196    0.2745
         0    1.0000    0.9333
    0.9294    0.4078    0.6431
    1.0000    0.6039    0.2314
    0.2275    0.1020    0.8471
    0.7882    0.0157    0.0157
    0.7294    0.6392    0.2275
    0.4314    0.4275    0.6706
    0.2235    0.1176    0.2824
    0.0314    0.6000    0.6078
    0.7882    0.2118    0.6078
    0.9608    0.4745    0.1059
    0.2000    0.5647    0.7569
    0.6588    0.2510    0.2510
    0.8706    0.6706    0.0706
    0.3216    0.2706    0.5098
    0.0196    0.6196    0.1490
    0.3412    0.9176    0.9804
    0.7686    0.5373    0.6471
    0.7804    0.4706    0.1412
    0.0980    0.0471    0.5804
    0.4510    0.1490    0.1490
    0.6510    0.5137    0.0706
    0.5686    0.2627    0.6118
    0.0196    0.8549    0.4588
    0.1529    0.3490    0.4784
    0.9608    0.7490    0.8863
    0.9255    0.6549    0.4353
    ];

% if you want to see these colors run
%
% >> figure, image(reshape(my_color_order, [], 1, 3)))
%

plot_parameters = {
    'defaultLineLineWidth',2,... makes lines in plots thicker by default
    'defaultScatterMarkerFaceColor' , 'flat',... *
    'defaultScatterMarkerEdgeColor' , 'none',... * makes scatter plot filled by default
    'defaultSurfaceEdgeAlpha' , 0,... surfaces will not have edges
    'defaultErrorbarLineWidth' , 1,...
    'defaultBarFaceColor' , 'flat'
    }';

axis_parameters = {
    'DefaultAxesFontSize',20,...
    'DefaultColorbarFontSize',20,...
    'DefaultAxesBox' , 0,...
    'DefaultAxesFontName' , 'Helvetica',...
    'DefaultAxesColorOrder', my_color_order,...
    'DefaultAxesLineWidth' , 1,...
    'DefaultAxesTickDir','out',... sets tick marks to outside
    'DefaultAxesLineWidth', 2
    }';

figure_parameters = {
    'defaultFigureColor','w',...
    'defaultFigureColormap', parula,...
    'defaultFigureToolbar','none',...
    'defaultFigureWindowButtonMotionFcn',''...
    }';

set(groot,...
        figure_parameters{:},...
        axis_parameters{:},...
        plot_parameters{:} ...
    );

end