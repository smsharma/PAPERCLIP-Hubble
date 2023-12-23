#!/bin/bash

#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
# #SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:4
#SBATCH --gres=gpu:4
#SBATCH --account=iaifi_lab
#SBATCH -p iaifi_gpu_priority

export TF_CPP_MIN_LOG_LEVEL="2"

# Load modules
module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01
module load gcc/12.2.0-fasrc01
module load openmpi/4.1.4-fasrc01

export ENV=multimodal-hubble

# Activate env
mamba activate $ENV

alias pip=/n/holystore01/LABS/iaifi_lab/Users/smsharma/envs/$ENV/bin/pip
alias python=/n/holystore01/LABS/iaifi_lab/Users/smsharma/envs/$ENV/bin/python
alias jupyter=/n/holystore01/LABS/iaifi_lab/Users/smsharma/envs/$ENV/bin/jupyter

cd /n/holystore01/LABS/iaifi_lab/Users/smsharma/multimodal-data/
/n/holystore01/LABS/iaifi_lab/Users/smsharma/envs/$ENV/bin/python -u train.py --config ./configs/base.py