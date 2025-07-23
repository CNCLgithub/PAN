function board_params = getGameBoardParamerters(varargin)
%%

% Parse optional arguments
% Defaults set standard game board to match rishi's and things like
% paddle dynamics and sensativity
p = inputParser;
addParameter(p,"paddle_pos"     , 'right')
addParameter(p,"right_left_dist", .2)
addParameter(p,"top_bottom_dist", .2)
addParameter(p,"sensativity"    , .002) % paddle sensativity
addParameter(p,"buffer"         , .01)   % buffer for collisions
addParameter(p,"paddle_exist"   , true, @islogical)

parse(p, varargin{:});

b.paddle_pos      = p.Results.paddle_pos;
b.right_left_dist = p.Results.right_left_dist;
b.top_bottom_dist = p.Results.top_bottom_dist;
b.sensativity     = p.Results.sensativity;
b.buffer          = p.Results.buffer;
b.paddle_exist    = p.Results.paddle_exist;

if ~b.paddle_exist, b.sensativity = 0.00001; end
% Calculate Board width and height
b.x_width = b.right_left_dist/2;
b.y_width = b.top_bottom_dist/2;

wall = [
    -1 -1
    -1 +1
    -1 +1
    +1 +1
    +1 +1
    +1 -1
    +1 -1
    -1 -1
    ].*[b.x_width b.y_width];


switch b.paddle_pos
    case 'left' % variable-dim = x ; fixed-dim = y
        wall(1:2,:) = nan; % set plotting wall with paddle to nans
        pxy         = -b.x_width;  % paddle fixed dimension value
        pvar_idx    = 4; % paddle variable dimension index
        pfix_idx    = 3; % paddle fixed dimension index
        pvarlim     = b.y_width; % paddle variable dimension limits
    case 'top' % variable-dim = y ; fixed-dim = x
        wall(3:4,:) = nan;
        pxy         = b.y_width; 
        pvar_idx    = 3; 
        pfix_idx    = 4; 
        pvarlim     = b.x_width;
    case 'right'
        wall(5:6,:) = nan;
        pxy         = b.x_width; 
        pvar_idx    = 4; 
        pfix_idx    = 3; 
        pvarlim     = b.y_width;
    case 'bottom'
        wall(7:8,:) = nan;
        pxy         = -b.y_width; 
        pvar_idx    = 3; 
        pfix_idx    = 4;
        pvarlim     = b.x_width;
end

% wall limits (for plotting)
b.plot_wall_x = wall(:,1);
b.plot_wall_y = wall(:,2);

% ratio between our hight/width and Rishi's Mpong parameters [-10 10]º 
b.rishi_conversion = b.top_bottom_dist/20; 

% Fixed/Variable Paddle information
b.paddle_var_index  = pvar_idx;
b.paddle_var_limits = pvarlim;
b.paddle_fix_index  = pfix_idx;
b.paddle_fix_value  = pxy;


% plot(b.plot_wall_x,b.plot_wall_y)
% axis equal
% set(gca,'XLim',[-1 1]*b.x_width*1.2,'YLim',[-1 1]*b.y_width*1.2);

board_params = b;

end

