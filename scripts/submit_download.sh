#!/bin/bash

#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=08:00:00
#SBATCH --account=iaifi_lab
#SBATCH --array=0-15%20 # Creates 8 jobs (32 cycles / 2 cycles per job), with at most 20 running simultaneously.

export TF_CPP_MIN_LOG_LEVEL="2"

# Load modules
module load python/3.10.9-fasrc01

# Activate env
mamba activate jax

# Go to dir
cd /n/holystore01/LABS/iaifi_lab/Users/smsharma/multimodal-data/

# Calculate cycle_min and cycle_max for each array job
CYCLE_PER_JOB=2
CYCLE_MIN=$(( SLURM_ARRAY_TASK_ID * CYCLE_PER_JOB ))
CYCLE_MAX=$(( (SLURM_ARRAY_TASK_ID + 1) * CYCLE_PER_JOB - 1 ))

# Download
python -u download_data.py --max_resolution 512 --cycle_min $CYCLE_MIN --cycle_max $CYCLE_MAX --n_max_images 20 --seed 42 --data_dir "./data/"
