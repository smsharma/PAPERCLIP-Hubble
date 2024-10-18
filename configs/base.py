import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # Wandb logging
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.entity = None
    wandb.project = "multimodal-hubble"
    wandb.group = "proposals"
    wandb.job_type = "training"
    wandb.name = None
    wandb.log_train = True
    wandb.workdir = "/n/holystore01/LABS/iaifi_lab/Users/smsharma/multimodal-data/logging/"

    # CLIP
    config.clip = clip = ml_collections.ConfigDict()
    clip.dtype = "float32"
    clip.use_pretrained = True
    clip.pretrained_model_name = "openai/clip-vit-base-patch16"  # "openai/clip-vit-large-patch14"
    clip.random_init_vision = False
    clip.random_init_text = False
    clip.transfer_head = False
    clip.d_transfer_head = 1024

    # Data
    config.data = data = ml_collections.ConfigDict()
    data.augment_rotate = True
    data.augment_rotate_type = "discrete"  # "continuous" rotation angles or "discrete" 90-deg flips
    data.augment_crop = True
    data.augment_subsample_text = False
    data.max_length_words = 77  # Max words used in subsampling
    data.tfrecords_dir = "tfrecords_v5"
    data.caption_type = "summary"  # "abstract" or "summary"
    data.shuffle_within_batch = False
    data.data_dir = "/n/holyscratch01/iaifi_lab/smsharma/hubble_data/"

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.half_precision = False
    training.train_fraction = 0.95
    training.batch_size = 32  # Must be divisible by number of devices; this is the total batch size, not per-device
    training.batch_size_val = 100
    training.n_train_steps = 20_001
    training.warmup_steps = 2000
    training.log_every_steps = 100
    training.eval_every_steps = 200
    training.loss_type = "softmax"  # "sigmoid" or "softmax"
    training.ckpt_best_metric = "loss"  # "loss" or "top_xx_accuracy"
    training.ckpt_best_metric_best_mode = "min"  # "max" or "min" 
    training.ckpt_keep_top_n = 3  # Save the top `ckpt_keep_top_n` checkpoints based on `ckpt_best_metric`
    training.load_ckpt = False
    training.ckpt_run_name = "scintillating-rat-118"


    # Sum1 options
    config.sum1 = sum1 = ml_collections.ConfigDict()
    sum1.use_sum1 = False
    sum1.summaries_filename = "summary_v2"
    sum1.sum1_filename = "summary_sum1_v3"

    # Optimizer (AdamW)
    config.optim = optim = ml_collections.ConfigDict()
    optim.schedule = "constant"
    optim.learning_rate = 1e-5
    optim.weight_decay = 1e-3

    # Seed
    config.seed = 42

    return config
