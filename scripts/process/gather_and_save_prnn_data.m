function gather_and_save_prnn_data(runids)
%%

runDate = "28-Feb";

runids = dir("../../hpc-outputs/programmed-networks/"+runDate+"*");

[mtp_prnns,rsa_prnns] = load_prnn_run_data(runids);


end


function [mtp_prnns,rsa_prnns] = load_prnn_run_data(runids)
   
    hpcdir = "../../hpc-outputs/programmed-networks";
    nruns = numel(runids);
    mtp_prnns = struct;
    rsa_prnns = struct;
    for j = 1:nruns
        printloop(j,nruns)
        rundir = runids(j).name;
       
        files  = dir(fullfile(hpcdir,rundir,"rnn-states/*.mat"));
        nfiles = numel(files);
    
        if nfiles == 0, continue,end
        
        runmtp = cell(nfiles,1);
        runrsa = cell(nfiles,1);
        parfor i = 1:nfiles
            tmp = load(fullfile(files(i).folder,files(i).name),"ftp_valid","rsaStruct");
            
            runmtp{i} = remove_beta_weights(tmp.ftp_valid);
            runrsa{i} = tmp.rsaStruct;
        end

        % Read the parameter log file
        file_content = fileread(fullfile(hpcdir,rundir,"parameter_log.txt"));
        
        % Use regular expression to find the number of neurons
        neurons_pattern = 'Number of neurons: (\d+)';
        neurons_match = regexp(file_content, neurons_pattern, 'tokens');
        
        % Extract the number of neurons
        nneur = str2double(neurons_match{1}{1});
        
        mtp_prnns.("neurons"+string(nneur)) = vertcat(runmtp{:});
        rsa_prnns.("neurons"+string(nneur)) = vertcat(runrsa{:});
    
    end

end


function mtp = remove_beta_weights(mtp)

    fields = string(fieldnames(mtp));

    for i = 1:numel(fields)

        mtp.(fields(i)) = rmfield(mtp.(fields(i)),"beta_weights");
        mtp.(fields(i)) = rmfield(mtp.(fields(i)),"optimal_lambdas");

    end



end
