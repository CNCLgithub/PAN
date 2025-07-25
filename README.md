# Programmed Attractor Networks (PAN)

Code repository for "Uncovering Neural Mechanisms of Mental Simulation With Programmed Attractor Networks"

## Overview

This repository contains the implementation of Programmed Attractor Networks (PAN), a novel multilevel computational theory that explains how the brain implements mental simulation through structure-preserving, physics-based representations. PANs bridge cognitive theories of mental simulation with neuroscience by embedding game-engine-style representations into recurrent neural networks via dynamical attractor manifolds.

Key features:
- Programs (no trainning) Reservoir Computing based RNNs with physics-based representations; this method was first introduced and formalized by JAson Kim and Dani Bassett in 2023. See [Kim & Bassett, 2023](https://www.nature.com/articles/s42256-023-00668-8) for the full methodology.
- Compares model activity to macaque dorsomedial frontal cortex (DMFC) recordings and to machine-learning style (task-optimized) RNNs.
- Decodes ball trajectories from biological and artificial networks hidden state-vectors to compare symbolic information within the recurrent latent population.
- Uses Representational Similarity Analysis (RSA) to test structure-preserving vs. task-optimized representations of physical scenes

## Project Structure

```
PAN/
├── data/                       # Neural recordings and ML-style model data from Jazayeri et al. (2025)
│   ├── brain_and_rishi_data.mat  # Combined neural and behavioral data
│   └── bbrnns.mat             # Task-optimized RNN models
├── scripts/
│   ├── process/               # Main analysis scripts
│   │   ├── create_and_program_pong_rnns.m  # Create PAN models
│   │   ├── run_rnn_trials.m               # Run simulations
│   │   ├── bounce_nobounce_crossDecoding.m # Cross-condition analysis
│   │   └── calculate_RDMs_for_neural_and_ML_models.m # RSA analysis
│   ├── plot-functions/        # Visualization tools
│   ├── dependencies/          # External dependencies
│   │   └── kim_and_bassett_modified/  # RNN programming framework
│   └── objects/               # Data structures
├── figures/                   # Figure generation scripts
├── job-scripts/              # HPC batch processing scripts
└── hpc-outputs/              # Results from HPC runs

```

## Installation

### Requirements
- MATLAB (tested on R2023b or later)
- Required MATLAB toolboxes:
  - Signal Processing Toolbox
  - Statistics and Machine Learning Toolbox
  - Optimization Toolbox
  - Parallel Computing Toolbox (for HPC runs)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/CNCLgithub/PAN.git
cd PAN
```

2. Initialize the MATLAB environment (I think I added this to the top of most of the scripts, but running this will ensure all paths are set correctly):
```matlab
% In MATLAB, navigate to the PAN directory and run:
setup_PAN_environment()
```

This will add all necessary paths and initialize the plotting environment. NOTE: I ran this on UNIX, so you may need to adjust the paths if you're on Windows, I tried to make the requisite changes in the scripts, but if you run into issues, this would be something to check.

## Quick Start
### Minimal Example: Create and Test One PAN Network Using the main script
The main script `train_and_run_rnn_worldmodels.m` provides a quick way to create and test a single PAN network. It automatically detects if running in MATLAB Live Editor and creates one network for debugging.

To run it use the MATLAB editor window and the code can be run in blocks sequentially.

### Minimal Example: Create and Test One PAN Network Calling the individual scripts

```matlab
% 1. Set up environment
setup_PAN_environment()

% 2. Create a single PAN network
cd scripts/process
prnn_info = create_and_program_pong_rnns(1, 1000, 'programmed', 'paddle_exist', false);

% 3. Load experimental data
load('../../data/brain_and_rishi_data.mat', 'rishi_data');

% 4. Run network on validation trials
prnn_out = run_rnn_trials(prnn_info.prnn, rishi_data);

% 5. Visualize a trial
prnn_out.plot_game_trial(1);  % Plot trial 1

% 6. Check ball endpoint decoding over time
% (See full pipeline in train_and_run_rnn_worldmodels.m)
```

## Core Workflow

### Step 1: Generate PAN Networks
The main entry point is `train_and_run_rnn_worldmodels.m`, which:
- Creates programmed RNNs with physics-based dynamics
- Runs them on validation trials (monkey experimental conditions)
- Performs initial decoding analyses
- Saves results as `prnnvars` objects


**HPC execution (multiple networks):**
```bash
# Generate jobs for 100 networks (default)
./job-scripts/main-write-jobs.sh --nmodels 100 --nneurons 1000

# Submit to SLURM cluster
sh dsq-caller.sh train-jobs.txt --time 1-00:00:00 --partition day
```

### Step 2: Post-hoc Analyses
After generating networks, run additional analyses on saved `prnnvars`:

```matlab
% Load saved network results
load('hpc-outputs/programmed-networks/[runid]/rnn-states/prnn-states-id_*.mat');

% Run RSA analysis
run scripts/process/static_RSA_post_hoc.m

% Run additional decoding analyses
run scripts/process/bounce_nobounce_crossDecoding.m

% Generate figures
run scripts/plot-functions/supplementary_figure_3_Latent_Error.m
```

## Key Data Structure: prnnvars

The `prnnvars` class is the central data structure containing:

### Properties
- **Network components:**
  - `progrnn`: The programmed RNN object
  - `W`: Programmed readout matrix mapping hidden states to symbolic board-state variables encoded in the programmed differential equations
  - `board_params`: Physical parameters (board size, wall positions, etc.)

- **States and data:**
  - `network_states`: Raw RNN hidden states for each trial
  - `binned_states`: States binned at 50ms (matching neural recordings)
  - `initial_states_raw/normed`: Starting conditions for each trial
  - `network_inputs/outputs`: Ball trajectories and paddle positions

- **Analysis results:**
  - `final_timepoint_prediction`: Decoding performance over time
  - `rdm_vectors_rsa`: Representational dissimilarity matrices

- **Metadata:**
  - `runid`, `netid`: Unique identifiers
  - `savedir`: Output directory path

### Key Methods
```matlab
% Re-run network to regenerate states (avoiding large file storage)
prnnvars_obj = prnnvars_obj.run_network();

% Visualize single trial
prnnvars_obj.plot_game_trial(trial_idx);

% Compute representational dissimilarity matrices
prnnvars_obj.compute_RDMs_rsa();
```

**Note:** Full network states are not saved to disk (would be ~GB per network). Instead, the `prnnvars` object contains the programmed RNN and initial conditions, allowing quick regeneration of states when needed.

## Reproducing Key Results

### Step 1: Generate PAN Networks
First, create a set of PAN networks:

```matlab
% For quick testing (single network locally)
cd scripts/process
edit train_and_run_rnn_worldmodels.m
% Run in Live Editor mode

% For full results (100 networks on HPC)
cd job-scripts
./main-write-jobs.sh --nmodels 100 --nneurons 1000
../dsq-caller.sh train-jobs.txt
```

### Step 2: Run Analyses

After networks are generated, run post-hoc analyses:

#### Figure 2: Rapid Ball End-Point Decoding
```matlab
% Shows that both DMFC and PAN encode the ball's final position
% immediately after trial onset (~250ms)

% Load PAN results
load('hpc-outputs/programmed-networks/[runid]/rnn-states/prnn-states-id_*.mat');

% Run decoding analysis for bounce vs. no-bounce conditions
cd scripts/process
run bounce_nobounce_crossDecoding.m
```

#### Figure 3: Full Trajectory Decoding
```matlab
% Demonstrates that the entire future trajectory is linearly decodable
% The decoding analysis is already included in train_and_run_rnn_worldmodels.m
% Access results from: prnn_validation.final_timepoint_prediction
```

#### Figure 4: Representational Similarity Analysis
```matlab
% Compare similarity structure across conditions and time
cd scripts/process
run static_RSA_post_hoc.m  % For saved networks
% or
run rsa_current_05_01_2025.m  % For full pipeline
```

#### Supplementary Figure 3: Network Size Analysis
```matlab
% Test stability with different numbers of hidden units
% First generate networks with different sizes:
./main-write-jobs.sh --nmodels 10 --nneurons 300
./main-write-jobs.sh --nmodels 10 --nneurons 500
./main-write-jobs.sh --nmodels 10 --nneurons 1000

% Then run stability analysis
cd scripts/plot-functions
run supplementary_figure_3_Latent_Error.m
```

## HPC Pipeline

The HPC pipeline enables parallel generation of multiple PAN networks:

### 1. Generate Job File
```bash
cd job-scripts
./main-write-jobs.sh [options]

# Options:
#   --nmodels, -m    Number of networks to create (default: 100)
#   --nneurons, -n   Hidden units per network (default: 1000)
#   --connectivity   Type: 'programmed' or 'random' (default: programmed)
#   --ntrials, -t    Number of trials to run (default: 40)

# Example: Create 50 networks with 500 neurons each
./main-write-jobs.sh --nmodels 50 --nneurons 500
```

This creates:
- A jobfile (`train-jobs.txt`) with one line per network
- Output directories in `hpc-outputs/[connectivity]-networks/[unique-id]/`
- A unique run ID saved to `.tmp_last_job_id`

### 2. Submit Jobs to SLURM
```bash
# Uses dSQ for efficient job distribution
../dsq-caller.sh train-jobs.txt [options]

# Options:
#   --time, -t       Wall time (default: 1-00:00:00)
#   --partition, -p  SLURM partition (default: day)
#   --cpus, -c       CPUs per task (default: 13)
#   --mem, -m        Memory in GB (default: auto-calculated)

# Example: Submit with 2-day time limit
../dsq-caller.sh train-jobs.txt --time 2-00:00:00
```

### 3. Monitor Progress
```bash
# Check job status
squeue -u $USER

# View output directory (printed by main-write-jobs.sh)
ls hpc-outputs/programmed-networks/[run-id]/
```

### HPC SYNC files
I added my person pipeline for pulling and pushing information to the HPC. You can look at the `sync.sh` script used to synchronize the local and HPC directories. It uses `rsync` to efficiently transfer files between the two locations. The `pull.sh` script is used to pull the latest results from the HPC with some input flags should you want to limit the pulled files. There might be some path changes you need to make before using these scripts, but they should be straightforward to adapt, and I tried to comment them well.

### 4. Retrieve Results
Results are saved in:
- `rnn-states/`: Individual network files (`prnn-states-id_*.mat`)
- `rnn-analysis/`: Analysis outputs
- `parameter_log.txt`: Run configuration

## Data Description

### Neural Recordings
- 1,889 DMFC neurons from 2 macaques (Monkey M & P)
- 79 unique ball trajectory conditions
- 50ms time bins
- Ball interception task with occlusion

### Model Types
1. **PAN Models**: Structure-preserving RNNs programmed with physics dynamics
2. **Task-Optimized RNNs**: Traditional RNNs trained to predict ball position. Note that these were trained and published by Rishi et al. (2025) and contain many more hyperparameter categories than were used in the corresponding manuscript. For the paper, we limited the analysis to the best hyperparameter set which were models trained with 40 hidden units and received Gabor Filtered inputs rather than raw pixel inputs.
3. **Linear Map Heuristic**: Simple baseline mapping initial conditions to endpoints

## Key Findings

1. **Rapid Prediction**: Both DMFC and PAN encode ball endpoints ~250ms after trial onset
2. **Full Trajectory Encoding**: The entire future ball trajectory is linearly decodable from early states
3. **Structure Preservation**: PAN better explains DMFC activity than task-optimized models
4. **Dynamical Attractors**: Neural computation implements physics through latent manifolds

## Citation

If you use this code in your research, please cite:

```bibtex
@article{calbick2025pan,
  title={Uncovering Neural Mechanisms of Mental Simulation With Programmed Attractor Networks},
  author={Calbick, Daniel and Kim, Jason Z and Sohn, Hansem and Yildirim, Ilker},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.XX.XX.XXXXXX}
}
```

## Contact
- Daniel Calbick: daniel.calbick@yale.edu

## Acknowledgments

- Mehrdad Jazayeri for sharing neural data
- Jason Z. Kim & Dani S. Bassett for the RNN programming framework
- Yale Center for Research Computing for HPC resources

## Troubleshooting

### Common Issues

Feel free to contact me if you run into any issues, but here are some common issues and their solutions:

1. **Path conflicts**: If you get warnings about conflicting function names:
   ```matlab
   which function_name  % Find the conflict
   rmpath('path/to/conflicting/file')  % Temporarily remove
   ```

2. **Memory errors on HPC**: Adjust memory allocation:
   ```bash
   ../dsq-caller.sh train-jobs.txt --mem 20  # 20GB per job
   ```

3. **Missing dependencies**: The code uses modified versions of Kim & Bassett's framework located in `scripts/dependencies/kim_and_bassett_modified/`

4. **Large file sizes**: Network states are regenerated on-demand rather than saved to avoid GB-sized files. Use the `run_network()` method of `prnnvars` objects.

5. **MATLAB version**: Tested on R2023a/b. Earlier versions may have compatibility issues with some functions.