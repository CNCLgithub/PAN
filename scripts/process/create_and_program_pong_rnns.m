function prnns= create_and_program_pong_rnns(nnetworks,nneurons,connectivity,varargin)
%% Create Base Rnns



% Parse optional arguments
p = inputParser;
addParameter(p,"save"        , false, @islogical) % Save this batch of networks if explicitly asked
addParameter(p,"parallel"    , false, @islogical)
addParameter(p,"paddle_exist",  true, @islogical)

parse(p, varargin{:});

savetrig     = p.Results.save;
parfor_trig  = p.Results.parallel;
paddle_exist = p.Results.paddle_exist;


runParams.n  = nneurons; % number of latent variables (#neurons)
runParams.dt = 0.01; % rnn simulation time-step (s)
runParams.connectivity = connectivity;

board_params = getGameBoardParamerters("paddle_exist",paddle_exist);

if nnetworks == 0
    prnns.prnn = [];
    prnns.board_params = board_params;
    return
end

prnns(1:nnetworks,1) = struct("prnn",[],"board_params",board_params);
 
if parfor_trig || nnetworks > 20

    parfor i = 1:nnetworks
    
        basernn = program_rnn(runParams, board_params);
        
        prnns(i).prnn = basernn;
    
    end
else

     for i = 1:nnetworks
    
        basernn = program_rnn(runParams, board_params);
        
        prnns(i).prnn = basernn;
    
    end

end

if savetrig

    runid      = string(java.util.UUID.randomUUID.toString);
    today      = string(datetime("today"));
    saveString = strcat("../../data/untrained-prnns/",today, "_rnn_id_", runid , ".mat");
    save(saveString,"prnns",'-v7.3')

end



end












