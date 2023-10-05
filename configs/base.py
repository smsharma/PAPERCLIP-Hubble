import ml_collections
import dataclasses


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
    wandb.workdir = "./logging/"

    # Text
    config.text_config = text_config = ml_collections.ConfigDict()
    text_config.dtype = "float32"
    text_config.activations =  ("gelu",)
    text_config.use_bias = False
    text_config.force_scale = False
    text_config.attention_dropout = 0.0
    text_config.mlp_dropout_rate = 0.0
    text_config.unroll = 100
    text_config.gradient_checkpointing = True
    text_config.eos_token_id = 49407
    text_config.vocab_size = 50000
    text_config.hidden_size = 512
    text_config.max_length = 300
    text_config.num_layers = 5
    text_config.use_rmsnorm = True
    text_config.ln_type = "preln"
    text_config.num_heads = 8
    text_config.position_embedding_type = "rotary"
    text_config.use_causal_mask = False
    text_config.mlp_dim = 1024

    # Vision
    config.vision_config = vision_config = ml_collections.ConfigDict()
    vision_config.position_embedding_type = "sincos2d"
    vision_config.dtype = "float32"
    vision_config.activations = ("gelu",)
    vision_config.use_bias = False
    vision_config.force_scale = False
    vision_config.attention_dropout = 0.0
    vision_config.mlp_dropout_rate = 0.0
    vision_config.unroll = 100
    vision_config.gradient_checkpointing = True
    vision_config.image_size = 512
    vision_config.hidden_size = 512
    vision_config.patch_size = 32
    vision_config.num_layers = 4
    vision_config.use_rmsnorm = True
    vision_config.ln_type = "preln"
    vision_config.num_heads = 4
    vision_config.use_causal_mask = False
    vision_config.mlp_dim = 1024

    # CLIP
    config.clip = clip = ml_collections.ConfigDict()
    clip.projection_dim = 512
    clip.logit_scale_init_value = 1.0
    clip.logit_bias_init_value = -10.0
    clip.dtype = "float32"
    clip.use_pretrained = True
    clip.pretrained_model_name = "openai/clip-vit-base-patch16"

    # Data
    config.data = data = ml_collections.ConfigDict()
    data.augment_rotate = True
    data.augment_crop = False
    data.augment_subsample_text = False
    data.max_length_words = 77

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.half_precision = False
    training.train_fraction = 0.95
    training.batch_size = 32  # Must be divisible by number of devices; this is the total batch size, not per-device
    training.n_train_steps = 101_000
    training.warmup_steps = 5_000
    training.log_every_steps = 100
    training.eval_every_steps = 200
    training.save_every_steps = 20_000

    # Optimizer (AdamW)
    config.optim = optim = ml_collections.ConfigDict()
    optim.learning_rate = 1e-4
    optim.weight_decay = 1e-4

    # Seed
    config.seed = 42

    return config
