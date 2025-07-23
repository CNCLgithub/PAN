function [x,dx,board_params] = get_symbolic_pong_logic()
%%
m = 13; %#ok<NASGU> % Number of outputs for base RNN

% Define symbolic dynamics
board_params = getGameBoardParamerters(); 

xo = 0.01;
xw = board_params.x_width - xo;% width for the arena
yw = board_params.y_width - xo;% height for the arena

wallid = {'right' 'left' 'top' 'bottom'};
collision_dynamics = table('Size',[4 2],'VariableTypes',repmat("string",1,2), ...
    'VariableNames',{'collision' 'srinput'},'RowNames',wallid);
nwalls = numel(wallid);

ps =  0.002; % paddle sensativity

syms t; assume(t,'real'); syms x(t) [m,1]; syms xc [m,1]; x = x(t); 
assume(x,'real'); assume(xc,'real');

% Phase bifrucation constants, makes x-intercept of the phase diagram
% x_dot(x) 0.1 or -0.1, depending on the regime
fixedpoint = 0.1;
b = .01; 
a = -(1 + b) / (fixedpoint^2); 

% Jason's parameters
% xf = 0.1; cx = 3/13; ax = -cx/(3*0.025^2);

% Velocities, dx(1:2)
xvel = x(1);
yvel = x(2);

% Positions, dx(3:4)
xpos = x(5)*xvel; % x-srlatch [-1 or 1] * xvelocity
ypos = x(7)*yvel; % y-srlatch [-1 or 1] * yvelocity

% External Paddle Input, dx(13)
paddle = x(13);

% Collision Logic:
% srlatches, dx(5:8) and collision detectors, dx(9:12)

for i = 1:nwalls

    wall = wallid{i};
    idx  = 9+(i-1);
   
    if strcmp(wall ,board_params.paddle_pos)       
         switch wall
             case "right" , x_dist =  x(3) - xw     ; y_dist = x(4) - paddle;
             case  "left" , x_dist =  x(3) + xw     ; y_dist = x(4) - paddle;
             case "top"   , x_dist =  x(3) - paddle ; y_dist = x(4) - yw;
             case "bottom", x_dist =  x(3) - paddle ; y_dist = x(4) + yw; 
         end

         mult_sr = 50; mult_col = 2000;
         ball_dist = ps - (x_dist^2 + y_dist^2);
    else
        mult_sr = 10; mult_col = 1000;
        switch wall
            case "right" , ball_dist =  x(3) + xw ;
            case  "left" , ball_dist =   xw  -x(3);
            case "top"   , ball_dist =  x(4) + yw  ;
            case "bottom", ball_dist =   yw  -x(4) ;    
        end
    end

    col_eqn = mult_col*( ball_dist*x(idx) - x(idx)^3 );
    sr_eqn  = mult_sr*x(idx)^2  - 2*fixedpoint;
    collision_dynamics{wall, "collision" } = string(col_eqn);
    collision_dynamics{wall,   "srinput" } =  sr_eqn;

end

% Reccurent dynamics of the NAND, dx(5:8), gates to set up pitchfork phase diagram
% and over all dynamic behavior
nand_reccurent = a*x(5:8).^3 + b*x(5:8) - fixedpoint;

% NAND-gate logic, dx(5:8), with input from collision and from the paired NAND in sr-latch 
% this value will shift the reccurent phase diagram of the system
nand_logic = [
    ( str2sym(collision_dynamics{"right" ,"srinput"}) * ( x(6) - fixedpoint ) )
    ( str2sym(collision_dynamics{"left"  ,"srinput"}) * ( x(5) - fixedpoint ) )
    ( str2sym(collision_dynamics{"top"   ,"srinput"}) * ( x(8) - fixedpoint ) )
    ( str2sym(collision_dynamics{"bottom","srinput"}) * ( x(7) - fixedpoint ) )
    ]./(2*fixedpoint);

% Connect the sr-latches, dx(5:8)
srx = [
    20*(nand_reccurent(1) + nand_logic(1))
    20*(nand_reccurent(2) + nand_logic(2))
    ];

sry = [
    20*(nand_reccurent(3) + nand_logic(3))
    20*(nand_reccurent(4) + nand_logic(4))
    ];


% Define complete state-update:
% each row will be added to the variable oneach subsequent time-step
dx = [0    % x( 1) d(dx), velocity (no 𝛥 , constant velosity)
      0    % x( 2) d(dy), velocity (no 𝛥 , constant velosity)
      xpos % x( 3)  x+dx, position (gets signed xvelocity and magnitude added to update position)
      ypos % x( 4)  y+dy, position (gets signed yvelocity and magnitude added to update position)
      srx  % x(5/6) srlatch for left/right wall collisions (flips sign of xvelocity)
      sry  % x(7/8) srlatch for top/bottom wall collisions (flips sign of yvelocity)
      str2sym(collision_dynamics{"right","collision"} ) % x( 9)
      str2sym(collision_dynamics{"left","collision"}  ) % x(10)
      str2sym(collision_dynamics{"top","collision"}   ) % x(11)
      str2sym(collision_dynamics{"bottom","collision"}) % x(12)
      0 % x(13) paddle (outside input, not folded into recurrent model dynamics)
      ];

% Jason's original dx matrix
% dx = [x(2)*.1;...1
%       20*(-.1 + (10*x(7)^2 -xf -xf)*( x(3) -xf)/(2*xf)  + cx*x(2)  + ax*x(2)^3);...2
%       20*(-.1 + (10*x(8)^2 -xf -xf)*( x(2) -xf)/(2*xf)  + cx*x(3)  + ax*x(3)^3);...3
%       x(5)*.1;...4
%       20*(-.1 + (10*x(9)^2 -xf -xf)*( x(6) -xf)/(2*xf)  + cx*x(5)  + ax*x(5)^3);...5
%       20*(-.1 + (50*x(10)^2 -xf -xf)*( x(5) -xf)/(2*xf)  + cx*x(6)  + ax*x(6)^3);...6
%       1000*(( x(1)-xw)*x(7) - x(7)^3);...7
%       1000*((-xw-x(1))*x(8) - x(8)^3);...8
%       1000*(( x(4)-yw)*x(9) - x(9)^3);...9
%       2000*((.002 - ( (x(1)-x(11) )^2+ (x(4)-yp)^2 ))*x(10) - x(10)^3);...10
%       0];


board_params.sensativity  = ps;
board_params.buffer       = xo;
end

