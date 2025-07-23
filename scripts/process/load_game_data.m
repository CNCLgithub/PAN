function [train_params, game_params, brain ] = ...
    load_game_data(datapath,board_params,ntrials,runDataFlag)


dfile = fullfile(datapath,"brain_and_rishi_data.mat");


if ~exist("runDataFlag","var"),runDataFlag = true; end

train_params = deal([]);


if ~exist(dfile,"file")
    [brain,game_params] = gather_brain_and_validation_data(datapath,board_params);
else
    load(dfile,"brain","rishi_data")
    game_params = rishi_data;
end

%% Set parameters and generate training trials

if runDataFlag
    ntr_bounce = ceil(ntrials*game_params.bounce_ratio );
    
    plottrig = false;
    min_trial_time = min(game_params.simulation_time);
    
    bounce_starts  = ...
        get_valid_pong_starts(ntr_bounce,...
        'plot',plottrig,'blims',[1 1],'tlim',min_trial_time);
    nonbounce_starts = ...
        get_valid_pong_starts(ntrials - ntr_bounce,...
        'plot',plottrig,'blims',[0 0],'tlim',min_trial_time);
    
    valid_starts = mat2cell( table2array(cat(1,bounce_starts,nonbounce_starts)),ntrials, ones(1, 4));
    
    [x0, y0, heading0, speed0] = valid_starts{:};
    
    
    [~,train_coors] = run_rishi_sim(valid_starts{:}, [0 1]);
    
    h0 = deg2rad(heading0);
    s0 = speed0*game_params.rishi_conversion;
    
    train_params.x0 = x0*game_params.rishi_conversion;
    train_params.y0 = y0*game_params.rishi_conversion;
    train_params.xdot0 = s0.*cos(h0);
    train_params.ydot0 = s0.*sin(h0);
    
    ball_pos_converted = cellfun(@(b) b*game_params.rishi_conversion,train_coors.(game_params.bvar ),UniformOutput=false);
    
    bnc = cellfun(@(y) numel(unique(sign(diff(y)))>1)>1,train_coors.y,'UniformOutput',true);

    train_params.sim_coordinates    = train_coors;
    train_params.ball_position      = ball_pos_converted;
    train_params.ball_pos_final     = cellfun(@(b) b(end),ball_pos_converted);
    train_params.simulation_time    = cellfun(@numel, train_coors.(game_params.bvar ));
    train_params.bounce             = bnc;

else, train_params = [];
end

end


function [brain,game_params] = gather_brain_and_validation_data(datapath,board_params)
    monkeys = {'mahler' , 'perle'};
    nmonk   = numel(monkeys);
    
    mfile = @(str) fullfile(datapath,[str '_dataset.mat']);
    load(mfile(monkeys{1}),monkeys{1});
    load(mfile(monkeys{2}),monkeys{2});
    
    mcell = cell(nmonk,1);
    for i = 1:nmonk
    
        mname = monkeys{i};    
        mdata.(mname) = eval(mname);
    
        mcell{i} = mdata.(mname).neural_responses_reliable;
    
    end
    
    trialmask = mdata.(mname).masks.occ.start_end_pad0;
    vismask   = mdata.(mname).masks.occ.start_occ_pad0;
    occmask   = mdata.(mname).masks.occ.occ_end_pad0;
    trialmask(isnan(trialmask)) = 0;
    vismask(isnan(vismask))     = 0;
    occmask(isnan(occmask))     = 0;
    
    brain.data     = cat(1 , mcell{:});
    brain.all_mask = trialmask;
    brain.vis_mask = vismask;
    brain.occ_mask = occmask;
    
    
    %%
    
    
    switch board_params.paddle_pos
        case {'top','bottom'},bvar = 'x';
        case {'left','right'},bvar = 'y';
    end
    
    meta         = mdata.(mname).meta;
    x0           = meta.ball_offset_x;
    y0           = meta.ball_offset_y;
    heading0     = meta.ball_heading;
    speed0       = meta.ball_speed;
    bounce_ratio = sum(meta.n_bounce)/numel(meta.n_bounce);
    
    [~,rishi_coors] = run_rishi_sim(x0,y0, heading0, speed0 , [0 1]);
    
    % Their coordinates go from -10º to 10º, which is a magnitude of 20º
    % So we get the normalized conversion from our board-representation, which
    % goes from -.05 to .05, by dividing the two magnitudes and thus having
    % converting from their representational space to ours
    rishi_conversion = board_params.right_left_dist/20;
    
    
    % Apply conversion to Rishi's init parameters
    game_params.x0    = x0*rishi_conversion;
    game_params.y0    = y0*rishi_conversion;
    
    % Their headings are in degrees where as we are working with raddians
    h0 = deg2rad(heading0);
    
    % They have 3 velocity parameters that they can apply via "speed" so
    % essentially slow, medium, and fast, which needs to be converted to our
    % domain and then applyed to get the x/y velocities
    s0 = speed0*rishi_conversion;
    game_params.xdot0 = s0.*cos(h0);
    game_params.ydot0 = s0.*sin(h0);
    
    % Save other relevant variables
    ball_pos_converted = cellfun(@(b) b*rishi_conversion,rishi_coors.(bvar),UniformOutput=false);
    game_params.sim_coordinates    = rishi_coors;
    game_params.ball_position      = ball_pos_converted;
    game_params.ball_pos_final     = cellfun(@(b) b(end),ball_pos_converted);
    game_params.simulation_time    = cellfun(@numel, rishi_coors.(bvar));
    game_params.bounce_ratio       = bounce_ratio;
    game_params.rishi_conversion   = rishi_conversion;
    game_params.bvar               = bvar;

end










