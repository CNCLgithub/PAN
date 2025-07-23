#!/bin/bash

# Set default values for dsq job submissions
time="1-00:00:00"
partition="day"
job_file_trigger=0


while (( "$#" )); do
  case "$1" in 
    --time|-t)
        shift ; time=$1 ;;
    --partition|-p)
        shift ; partition=$1 ;;
    --cpus|-c)
        shift ; CPUS_PER_TASK=$1 ;;
    --mem|-m)
    shift ; MEMORY_ALLOCATION=${1}g ;;
    *)
        if [ $job_file_trigger -eq 0 ]; then
            jobfile_name=$1
            job_file_trigger=1
            if [ -z "$jobfile_name" ]; then
                echo "Usage: dsq-caller.sh <jobfile_name> <flags>"
                exit 1
            fi
        else
            echo "Error: Invalid argument $1" >&2 ; exit 1
        fi
        ;;
  esac
  shift
done

# Constants
TOTAL_CPUS_PER_USER=128
TOTAL_USER_MEMORY_GB=1280

if [ -z "$CPUS_PER_TASK" ]; then 
    CPUS_PER_TASK=13
fi


if [ -z "$MEMORY_ALLOCATION" ]; then 
    # Calculate maximum number of concurrent tasks
    MAX_TASKS=$(($TOTAL_CPUS_PER_USER / $CPUS_PER_TASK))

    # Calculate memory per task and floor the result
    TOTAL_PAR_CPUS=$(echo "scale=2; $MAX_TASKS*$CPUS_PER_TASK " | bc)
    MEMORY_PER_TASK=$(echo "scale=2; $TOTAL_USER_MEMORY_GB / $TOTAL_PAR_CPUS" | bc)
    FLOORED_MEMORY_PER_TASK=$(echo "($MEMORY_PER_TASK)/1" | bc)

    # Store the floored memory amount into a variable
    MEMORY_ALLOCATION=${FLOORED_MEMORY_PER_TASK}g
fi



/home/dc938/software/dSQ/dsq --job-file $jobfile_name --nodes=1 --tasks=$CPUS_PER_TASK --cpus-per-task=1 --mem-per-cpu=$MEMORY_ALLOCATION  --partition=$partition -t $time --submit 
