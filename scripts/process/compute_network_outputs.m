function compute_network_outputs(stateFile)

    rootdir = '../..';

    addpath(genpath(rootdir))
    disp(stateFile)
    % mfile_path = fullfile(rootdir,stateFile);
    mfile_path = stateFile;
    mfile      = matfile(mfile_path);
    
    prnn = mfile.network_structures;
    
    pause(randi(50))

    prnn.run_networks();
    
    nets = prnn.rnnvars_valid2;
    
%%
    prnn_outputs = cellfun(@(x) x.network_outputs(),nets,'UniformOutput',false);
    
    svName = replace(mfile_path,"states","outputs");
    % svName = [rootdir '/test_output.mat'];
    save(svName,"prnn_outputs",'-v7.3')




end