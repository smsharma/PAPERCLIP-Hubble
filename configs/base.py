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
    wandb.log_train = False
    wandb.workdir = "/n/holystore01/LABS/iaifi_lab/Users/smsharma/multimodal-data/logging/"

    # CLIP
    config.clip = clip = ml_collections.ConfigDict()
    clip.dtype = "float32"
    clip.use_pretrained = True
    clip.pretrained_model_name ="openai/clip-vit-base-patch16"  # "openai/clip-vit-large-patch14"
    clip.random_init_vision = False
    clip.random_init_text = False

    # Data
    config.data = data = ml_collections.ConfigDict()
    data.augment_rotate = True
    data.augment_crop = True
    data.augment_subsample_text = True
    data.max_length_words = 77
    data.tfrecords_dir = "tfrecords_v3"
    data.shuffle_within_batch = False

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.half_precision = False
    training.train_fraction = 0.95
    training.batch_size = 32  # Must be divisible by number of devices; this is the total batch size, not per-device
    training.batch_size_val = 100
    training.n_train_steps = 100_001
    training.warmup_steps = 5000
    training.log_every_steps = 100
    training.eval_every_steps = 200
    training.loss_type = "softmax"  # "sigmoid" or "softmax"
    training.n_eval_batches = 10  # How many batches to use for evaluation
    training.ckpt_best_metric = ("top_10_accuracy", "loss")
    training.ckpt_best_metric_best_mode = ("max", "min")  # "max" or "min" for each metric in `ckpt_best_metric`
    training.ckpt_keep_top_n = 3  # Save the top `ckpt_keep_top_n` checkpoints based on `ckpt_best_metric`

    # Optimizer (AdamW)
    config.optim = optim = ml_collections.ConfigDict()
    optim.schedule = "cosine"
    optim.learning_rate = 1e-5
    optim.weight_decay = 1e-3

    # Seed
    config.seed = 42

    return config
