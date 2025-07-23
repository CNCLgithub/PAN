#!/bin/zsh

#  --exclude='hpc-outputs/'\

caffeinate rsync -avh --progress \
 --exclude='.*'\
 --exclude='*.txt' \
 --exclude='figures/' \
 --exclude='archive/'\
 --exclude={'pull.sh','sync.sh'} \
 --exclude='data/phys/' \
 --exclude='data/hansem_mat_files/' \
 --exclude='data/for_symbolicRNN/offline_rnn_neural_responses_reliable_50_occ.pkl' \
 --exclude='data/archive/' \
 --exclude='data/' \
 --exclude='RSA-example/' \
. dc938@misha.ycrc.yale.edu:/gpfs/radev/project/yildirim/dc938/world-models