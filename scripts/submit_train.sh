#!/bin/bash

#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --mem=200GB
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

export ENV=multimodal-hubble

# Activate env
mamba activate $ENV

alias pip=/n/holystore01/LABS/iaifi_lab/Users/smsharma/envs/$ENV/bin/pip
alias python=/n/holystore01/LABS/iaifi_lab/Users/smsharma/envs/$ENV/bin/python
alias jupyter=/n/holystore01/LABS/iaifi_lab/Users/smsharma/envs/$ENV/bin/jupyter

cd /n/holystore01/LABS/iaifi_lab/Users/smsharma/multimodal-data/

# Core runs

# /n/holystore01/LABS/iaifi_lab/Users/smsharma/envs/$ENV/bin/python -u train.py --config ./configs/base.py --config.data.caption_type="summary"  # Base config, with summary captions
# /n/holystore01/LABS/iaifi_lab/Users/smsharma/envs/$ENV/bin/python -u train.py --config ./configs/base.py --config.clip.transfer_head=True  # Fine-tune just head

# /n/holystore01/LABS/iaifi_lab/Users/smsharma/envs/$ENV/bin/python -u train.py --config ./configs/base.py --config.data.caption_type="abstract"  --config.data.augment_subsample_text=True  # Full abstract
# /n/holystore01/LABS/iaifi_lab/Users/smsharma/envs/$ENV/bin/python -u train.py --config ./configs/base.py --config.clip.random_init_text=True --config.clip.random_init_vision=True  # From scratch

# /n/holystore01/LABS/iaifi_lab/Users/smsharma/envs/$ENV/bin/python -u train.py --config ./configs/base.py --config.data.shuffle_within_batch=True  # Shuffle within batch
# /n/holystore01/LABS/iaifi_lab/Users/smsharma/envs/$ENV/bin/python -u train.py --config ./configs/base.py  --config.sum1.use_sum1=True --config.sum1.sum1_filename="summary_sum1_v3" # Base config, with summary captions

# Additional runs

# /n/holystore01/LABS/iaifi_lab/Users/smsharma/envs/$ENV/bin/python -u train.py --config ./configs/base.py --config.optim.learning_rate=1e-6  # Lower LR
# /n/holystore01/LABS/iaifi_lab/Users/smsharma/envs/$ENV/bin/python -u train.py --config ./configs/base.py --config.optim.schedule="cosine" # Cosine lr schedule
# /n/holystore01/LABS/iaifi_lab/Users/smsharma/envs/$ENV/bin/python -u train.py --config ./configs/base.py --config.training.loss_type="sigmoid"
# /n/holystore01/LABS/iaifi_lab/Users/smsharma/envs/$ENV/bin/python -u train.py --config ./configs/base.py --config.training.batch_size=64  # Larger batch size
# /n/holystore01/LABS/iaifi_lab/Users/smsharma/envs/$ENV/bin/python -u train.py --config ./configs/base.py  --config.training.n_train_steps=5000 --config.optim.learning_rate=1e-6  # Train for shorter
# /n/holystore01/LABS/iaifi_lab/Users/smsharma/envs/$ENV/bin/python -u train.py --config ./configs/base.py --config.clip.pretrained_model_name="openai/clip-vit-large-patch14"  # Larger base model
