classdef prnnvars < matlab.System

    properties
        progrnn
        data
        board_params

        start_states

        initial_states_raw
        network_states_raw
        initial_states_normed
        network_states_normed

        network_states
        binned_states
        binned_states_raw

        trial_mask
        input_data

        rdm_vectors_rsa
        final_timepoint_prediction

        W
        network_inputs
        network_outputs

        runid
        netid
        runOrdinal
        savedir
        daterun

    end

    
    methods 
        % Constructor
        function obj = prnnvars()
            obj.network_outputs = [];
            obj.daterun = datetime("now");
        end
        
        function varargout = plot_game_trial(self, idx , ax)

            if ~exist("ax","var"),ax = self.figure_setup();end
            self.plot_game(ax,idx);

            varargout{1} = ax;

        end

        function varargout = plot_trial_set(self)

            if ~exist("ax","var"),ax = self.figure_setup();end
            self.plot_set(ax);
            varargout{1} = ax;

        end

        function compute_RDMs_rsa(self)

            disttype = "euclidean";
    
            disp("Computing RDM Vectors")
                             
            self.rdm_vectors_rsa = self.compute_RDM_vectors( ...
                self.binned_states,disttype);
            
        end
        
        function self = run_network(self)
            self = run_rnn_trials(self.progrnn.basernn, self.network_inputs,"parallel",true,"printStuff",false);
            % self.network_states = prnn.network_states;
        end


    end

    methods (Access=private)

        function ax = figure_setup(self)
        
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
            
            ud.outputs = self.network_outputs;
            if isempty(ud.outputs)
                ud.outputs = self.get_network_outputs();
            end
            
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
            hold(ax,'off')

            ax.UserData = ud;


        end

        function plot_game(~,ax,idx)

            ud     = ax.UserData; 

            b      = ud.outputs{idx};         
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

        function plot_set(~,ax)

            ud = ax.UserData; 

            outputs = ud.outputs;
            ntrials = numel(outputs);

            ball   = findall(ax,'-depth',1,'Tag','ball');
            ball0  = findall(ax,'-depth',1,'Tag','ball_t0');

            ballc  = turbo(ntrials);

            for i = 1:ntrials

                b      = ud.outputs{i};         
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

        function out = get_network_outputs(self)

            % states = self.network_states;
            out = cellfun(@(s) self.W*s, self.network_states,uniformoutput=false);

        end

        % RSA Functions
        function residualVec = regress_out(targetVec, regressors)

            %% ========================================================================
            % This function removes the linear contribution of one or more regressor
            % vectors from a target vector by ordinary least squares. 
            %
            % Usage:
            %   dBrain_res = regress_out(dBrain, dReg);
            %   dBrain_res = regress_out(dBrain, [dReg1, dReg2, ...]);
            %
            % Inputs:
            %   - targetVec: (N x 1) or (1 x N) vector of distances (e.g., from the brain)
            %   - regressors: (N x k) matrix (or (1 x N) for single regressor) 
            %                 where each column is a regressor distance vector.
            %
            % Output:
            %   - residualVec: same size as targetVec, the residual after regressing out
            %                  all columns in `regressors`.
            %
            % The formula is: residual = targetVec - X * (X \ targetVec),
            % where X = [ones(N,1), regressors].
            %% ========================================================================
                % Ensure column vectors
                targetVec = targetVec(:);  
                
            
                % If there are no regressors, just return targetVec
                if isempty(regressors), residualVec = targetVec;return;end
            
                % Check that sizes match
                if size(regressors,1) ~= length(targetVec)
                    error('Target and regressors must have the same number of rows (trials).');
                end
            
                % Build design matrix X: first column is intercept (all ones),
                % then each column is one regressor
                X = [ones(size(regressors,1),1), regressors];
            
                % Solve for betas: b = X \ targetVec  (least squares)
                b = X \ targetVec;
            
                % Predicted = X*b
                predicted = X * b;
            
                % Residual = targetVec - predicted
                residualVec = targetVec - predicted;
            
            end

        function rdm_vectors_by_time = compute_RDM_vectors(self,disttype)

            stateCell   = self.binned_states.modelStates;

            ntrials = numel(stateCell);
            tbins   = cellfun(@(x) size(x,2),stateCell);
            
            ntbins_max = max(tbins);
            
            % Different Trials last different amounts of time so we want to normalize
            % them by padding with nans to combine into one 3D matrix [neurons x time x trial] 
            for i = 1:ntrials
            
                pad = ntbins_max - tbins(i);
                st  = tbins(i)+1;
                fn  = st+(pad-1);
            
                stateCell{i}(:,st:fn) = nan;
            
            end
            
            stateMatrix = cat(3,stateCell{:});

            rdm_vectors_by_time = cell(ntbins_max,1);
            for i = 1:ntbins_max
                   
                state  = squeeze(stateMatrix(:,i,:))';
                nancol = isnan(state(:,1));
            
                state(nancol,:) = [];
            
                % This is how I used to do it but I decided to just use the vector form
                % derectly output by pdist; It seems the only difference is the
                % interleaving of the upertriangular matrix into a vector, which
                % shouldn't matter for the correlation as long as the vectors are
                % computed in the same way.
                % % % state_dist = squareform(pdist( state , disttype));   
                % % % state_dist = triu(state_dist,1);    
                % % % dstate     = state_dist(state_dist~=0);
            
                rdm_vectors_by_time{i} = pdist( state,disttype)';
            
            end
            
            
              
        end



    end
    
    

    

end