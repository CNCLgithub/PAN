#!/bin/bash

# Initialize variables with defaults
netid=""
runid=""
connectivity="programmed"
nneurons=100
ntrials=40

while (( "$#" )); do
    case "$1" in
        --netid|-n)
            shift ; netid="$1" ;;
        --runid|-r)
            shift ; runid="$1" ;;
        --connectivity|-c)
            shift ; connectivity="$1" ;;
        --nneurons|-m)
            shift ; nneurons="$1" ;;
        --ntrials|-t)
            shift ; ntrials="$1" ;;
        *)
            echo "Error: Invalid argument $1" >&2
            exit 1
            ;;
    esac
    shift
done

# Validate required parameters
if [ -z "$netid" ] || [ -z "$runid" ]; then
    echo "Error: netid and runid are required parameters" >&2
    exit 1
fi

script_dir=scripts/process
cd "$script_dir" || exit 1

module load MATLAB/2023a
matlab -nosplash -nodisplay -r "train_and_run_rnn_worldmodels( \
                $netid, '$runid', \
                'nneurons', $nneurons, \
                'ntrials', $ntrials, \
                'connectivity', '$connectivity'); exit;"