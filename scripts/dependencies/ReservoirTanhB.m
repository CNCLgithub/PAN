classdef ReservoirTanhB < handle
    properties
        % Matrices
        A               % N x N matrix of internal reservoir connections
        B               % N x M matrix of s dynamical inputs to learn
        R               % N x N matrix of autonomous reservoir connections
        US
        V
        W
        % States and fixed points
        r               % N x 1 vector of current state
        rs              % N x 1 vector of reservoir fixed point
        xs              % s x 1 vector of input fixed point
        d               % N x 1 vector of bias terms
        % RK4 variables
        k1
        k2
        k3
        k4
        % Time
        delT            % Timescale of simulation
        gam             % Gamma: Timescale of reservoir evolution speed
    end
    
    methods
        % Constructor
        function obj = ReservoirTanhB(A, B, rs, xs, delT, gam)
            % Matrices
            obj.A = A;
            obj.B = B;
            % States and fixed points
            obj.rs = rs;
            obj.xs = xs;
            obj.d = atanh(rs) - A*rs - B*xs;
            % Time
            obj.delT = delT;
            obj.gam  = gam;
            % Initialize reservoir states to 0
            obj.r = zeros(size(A,1),1);
        end
        
        % Training: input both inputs x and control c 
        function D = train(o, x)

            nInd   = 0;  % Counter
            nx     = size(x,2); 

            D      = zeros(size(o.A,1), nx); % [neurons x time]
            D(:,1) = o.r;

            fprintf([repmat('.', [1, 100]) '\n']);
            for i = 2:nx
                if(i > nInd*nx); fprintf('='); nInd = nInd + .01; end
                o.propagate(x(:,i-1,:)); % Propagate States
                D(:,i) = o.r; % Save current state
            end
            fprintf('\n');
        end
        
        % Training: input both inputs x and control c 
        function D = trainSVD(o, x, k)
            [U,S,V] = svds(o.A,k);
            o.US = U*S;
            o.V = V';

            nInd = 0; nx = size(x,2);                 % Counter
            D = zeros(size(o.A,1), nx);
            D(:,1) = o.r;
            fprintf([repmat('.', [1, 100]) '\n']);
            for i = 2:nx
                if(i > nInd*nx); fprintf('='); nInd = nInd + .01; end
                o.propagateSVD(x(:,i-1,:)); % Propagate States
                D(:,i) = o.r;
            end
            fprintf('\n');
        end
        
        % Training: input both inputs x and control c 
        function D = train4(o, x)
            nInd = 0; nx = size(x,2);                 % Counter
            D = zeros(size(o.A,1), nx, 4);
            D(:,1,1) = o.r;
            fprintf([repmat('.', [1, 100]) '\n']);
            for i = 2:nx
                if(i > nInd*nx); fprintf('='); nInd = nInd + .01; end
                o.propagate(x(:,i-1,:)); % Propagate States
                D(:,i) = o.r;
                D(:,i-1,2:4) = D(:,i-1,1)+[o.k1/2 o.k2/2 o.k3];
            end
            fprintf('\n');
        end

        % Prediction
        function D = predict(o, W, nc)
            nInd = 0;
            o.R = o.A + o.B*W;                      % Feedback
            o.W = W;
            D = zeros(size(o.R,1), nc);
            D(:,1) = o.r;
            fprintf([repmat('.', [1, 100]) '\n']);
            for i = 2:nc
                if(i > nInd*nc); fprintf('='); nInd = nInd + .01; end
                o.propagate_x();            % Propagate States
                D(:,i) = o.r;
            end
            fprintf('\n');
        end

        % Prediction
        function D = predictSVD(o, W, nc, k)
            % Decompose
            [U,S,V] = svds(o.A + o.B*W, k);
            o.US = U*S;
            o.V = V';
            % Integrate
            nInd = 0;
            D = zeros(size(o.A,1), nc);
            D(:,1) = o.r;
            fprintf([repmat('.', [1, 100]) '\n']);
            for i = 2:nc
                if(i > nInd*nc); fprintf('='); nInd = nInd + .01; end
                o.propagateSVD_x();            % Propagate States
                D(:,i) = o.r;
            end
            fprintf('\n');
        end

        % Prediction
        function D = predict4(o, W, nc)
            nInd = 0;
            o.R = o.A + o.B*W;                      % Feedback
            o.W = W;
            D = zeros(size(o.R,1), nc, 4);
            D(:,1,1) = o.r;
            fprintf([repmat('.', [1, 100]) '\n']);
            for i = 2:nc
                if(i > nInd*nc); fprintf('='); nInd = nInd + .01; end
                o.propagate_x();            % Propagate States
                D(:,i) = o.r;
                D(:,i-1,2:4) = D(:,i-1,1)+[o.k1/2 o.k2/2 o.k3];
            end
            fprintf('\n');
        end
        
        
        %% RK4 integrator
        % driven reservoir
        function propagate(o,x)
            o.k1 = o.delT * o.del_r(o.r         , x(:,1,1));
            o.k2 = o.delT * o.del_r(o.r + o.k1/2, x(:,1,2));
            o.k3 = o.delT * o.del_r(o.r + o.k2/2, x(:,1,3));
            o.k4 = o.delT * o.del_r(o.r + o.k3  , x(:,1,4));
            o.r = o.r + (o.k1 + 2*o.k2 + 2*o.k3 + o.k4)/6;
        end

        % feedback reservoir
        function propagate_x(o)
            o.k1 = o.delT * o.del_r_x(o.r);
            o.k2 = o.delT * o.del_r_x(o.r + o.k1/2);
            o.k3 = o.delT * o.del_r_x(o.r + o.k2/2);
            o.k4 = o.delT * o.del_r_x(o.r + o.k3);
            o.r = o.r + (o.k1 + 2*o.k2 + 2*o.k3 + o.k4)/6;
        end

        % driven reservoirSVD
        function propagateSVD(o,x)
            o.k1 = o.delT * o.delSVD_r(o.r,        x(:,1,1));
            o.k2 = o.delT * o.delSVD_r(o.r + o.k1/2, x(:,1,2));
            o.k3 = o.delT * o.delSVD_r(o.r + o.k2/2, x(:,1,3));
            o.k4 = o.delT * o.delSVD_r(o.r + o.k3,   x(:,1,4));
            o.r = o.r + (o.k1 + 2*o.k2 + 2*o.k3 + o.k4)/6;
        end

        % feedback reservoirSVD
        function propagateSVD_x(o)
            o.k1 = o.delT * o.delSVD_r_x(o.r);
            o.k2 = o.delT * o.delSVD_r_x(o.r + o.k1/2);
            o.k3 = o.delT * o.delSVD_r_x(o.r + o.k2/2);
            o.k4 = o.delT * o.delSVD_r_x(o.r + o.k3);
            o.r = o.r + (o.k1 + 2*o.k2 + 2*o.k3 + o.k4)/6;
        end
        
        
        %% ODEs
        % driven reservoir
        function dr = del_r(o,r,x)
            dr = o.gam * (-r + tanh(o.A*r + o.B*x + o.d));
        end
        % feedback reservoir
        function dr = del_r_x(o,r)
            dr = o.gam * (-r + tanh(o.A*r + o.B*(o.W*r)+ o.d));
        end
        % driven reservoirSVD
        function dr = delSVD_r(o,r,x)
            dr = o.gam * (-r + tanh(o.US*(o.V*r) + o.B*x + o.d));
        end
        % feedback reservoirSVD
        function dr = delSVD_r_x(o,r)
            dr = o.gam * (-r + tanh(o.US*(o.V*r) + o.d));
        end
    end
end