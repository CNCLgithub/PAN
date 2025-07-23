#!/bin/bash

# Default settings
USER=dc938
CLUSTER_IP=misha.ycrc.yale.edu
HPC_MAIN_DIRECTORY=/gpfs/radev/project/yildirim/dc938/world-models/
LOCALDIR="./"
DRY_RUN=""

# Function to display usage information
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -n, --dry-run    Perform a trial run with no changes made"
    echo "  -d, --dir        Input a different directory to sync from (default: $HPC_MAIN_DIRECTORY)"
    echo "  -h, --help       Display this help message"
    exit 1
}

# Process command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--dry-run)
            DRY_RUN="-n"
            shift
            ;;
        -d|--dir)
            shift
            SUB_DIR="$1"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Execute rsync command
if [ -n "$DRY_RUN" ]; then
    echo "Performing dry run (no files will be transferred)..."
fi

if [ -n "$SUB_DIR" ]; then
    HPC_MAIN_DIRECTORY="$HPC_MAIN_DIRECTORY$SUB_DIR"
    LOCALDIR="$LOCALDIR$SUB_DIR"
    echo "Syncing from $HPC_MAIN_DIRECTORY to $LOCALDIR"
fi

caffeinate rsync -avhP $DRY_RUN \
  --exclude='.*' \
  --exclude='*jobs*.txt' \
  --exclude='*.out' \
  --exclude='*.tsv' \
  --exclude='*.pdf' \
  --exclude='hpc-outputs/programmed-networks/14-Feb*' \
  $USER@$CLUSTER_IP:$HPC_MAIN_DIRECTORY $LOCALDIR

#   --exclude='scripts/' \
