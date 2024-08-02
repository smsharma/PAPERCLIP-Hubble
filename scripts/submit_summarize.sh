#!/bin/bash

#SBATCH --job-name=summarize_mixtral
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=iaifi_lab
#SBATCH -p iaifi_gpu_priority

export TF_CPP_MIN_LOG_LEVEL="2"

# Load modules
module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01
module load gcc/12.2.0-fasrc01
module load openmpi/4.1.4-fasrc01

export ENV=outlines

# Activate env
mamba activate $ENV

alias pip=/n/holystore01/LABS/iaifi_lab/Users/smsharma/envs/$ENV/bin/pip
alias python=/n/holystore01/LABS/iaifi_lab/Users/smsharma/envs/$ENV/bin/python
alias jupyter=/n/holystore01/LABS/iaifi_lab/Users/smsharma/envs/$ENV/bin/jupyter

cd /n/holystore01/LABS/iaifi_lab/Users/smsharma/multimodal-data/
/n/holystore01/LABS/iaifi_lab/Users/smsharma/envs/$ENV/bin/python summarize.py --data_dir /n/holystore01/LABS/iaifi_lab/Users/smsharma/multimodal-data/data --observations_dir observations_v1 --save_filename summary_v2 --verbose