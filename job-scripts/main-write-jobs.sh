#!/bin/bash



# Set default values for dsq job submissions
caller_function=job-scripts/train-networks.sh
jobfile_name=train-jobs.txt
connectivity="programmed"
nmodels=100  
ntrials=40
nneurons=1000  

# Get a unique id for this job
uid="$(date +%d-%b-%Y_)${nneurons}neurons_$(cat /proc/sys/kernel/random/uuid)"

# Save the job id to a file
echo "$uid" > ".tmp_last_job_id"

while (( "$#" )); do
    case "$1" in
        --nneurons|-n)
            shift ; nneurons="$1" ;;
        --connectivity|-c)
            shift ; connectivity="$1" ;;
        --nmodels|-m)
            shift ; nmodels="$1" ;;
        --ntrials|-t)
            shift ; ntrials="$1" ;;
        -jobfile|-j)
            shift ; jobfile_name="$1" ;;
        *) break ;;
    esac
    shift
done

# Remove the jobfile if it exists.
if [ -f "$jobfile_name" ]; then rm "$jobfile_name"; fi

runid=$uid

# Create the jobs.
for i in $(seq 1 $nmodels); do
    echo "sh $caller_function \
                --netid $i \
                --runid $runid \
                --connectivity $connectivity \
                --nneurons $nneurons \
                --ntrials $ntrials" >> "$jobfile_name"
done

# Echo the output directory
con_fldr="${connectivity}-networks"
echo "hpc-outputs/$con_fldr/$runid"

# Create the output directory for this run
mkdir -p "hpc-outputs/$con_fldr/$runid/rnn-analysis"
mkdir -p "hpc-outputs/$con_fldr/$runid/rnn-states"