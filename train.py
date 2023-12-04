import sys
import os
import yaml
from functools import partial

sys.path.append("./")
sys.path.append("../")

from absl import flags, logging
from absl import logging
from ml_collections import config_flags, ConfigDict
from clu import metric_writers
import wandb

import jax
import jax.numpy as np
import optax
import flax
import orbax
from flax.core import FrozenDict
from flax.training import common_utils, train_state, orbax_utils

import tensorflow as tf
from dm_pix import rotate, random_crop
from tqdm import trange

from transformers import AutoTokenizer, AutoProcessor, FlaxCLIPModel

from models.clip import CLIPModel
from models.dataset_utils import make_dataloader
from models.train_utils import param_count, train_step, eval_step, to_wandb_config
from models.text_utils import process_truncate_captions, tokenize_captions

replicate = flax.jax_utils.replicate
unreplicate = flax.jax_utils.unreplicate

logging.set_verbosity(logging.INFO)

def train(config: ConfigDict, workdir: str = "./logging/") -> train_state.TrainState:
    # Set up wandb run
    if config.wandb.log_train and jax.process_index() == 0:
        wandb_config = to_wandb_config(config)
        run = wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            job_type=config.wandb.job_type,
            group=config.wandb.group,
            config=wandb_config,
        )
        wandb.define_metric("*", step_metric="train/step")  # Set default x-axis as 'train/step'
        workdir = os.path.join(workdir, run.group, run.name)

        # Recursively create workdir
        os.makedirs(workdir, exist_ok=True)

    # Devices
    num_local_devices = jax.local_device_count()
    num_hosts = jax.process_count()

    # Dump a yaml config file in the output directory
    with open(os.path.join(workdir, "config.yaml"), "w") as f:
        yaml.dump(config.to_dict(), f)

    writer = metric_writers.create_default_writer(logdir=workdir, just_logging=jax.process_index() != 0)

    # Set up data

    # Find all TFRecord files and make datasets
    files_train = tf.io.gfile.glob(f"./data/{config.data.tfrecords_dir}/*train*.tfrecord")
    train_ds = make_dataloader(files_train, batch_size=config.training.batch_size, seed=config.seed, split='train', shuffle=True)

    files_val = tf.io.gfile.glob(f"./data/{config.data.tfrecords_dir}/*val*.tfrecord")
    val_ds = make_dataloader(files_val, batch_size=config.training.batch_size_val, seed=config.seed, split='val', shuffle=False)

    batches = iter(train_ds)

    logging.info("Loaded the dataset")

    # Model configs
    text_config = FrozenDict(config.text_config)
    vision_config = FrozenDict(config.vision_config)
    clip_config = FrozenDict(config.clip)

    # Use pre-trained model or train from scratch
    if config.clip.use_pretrained:
        model = FlaxCLIPModel.from_pretrained(config.clip.pretrained_model_name, dtype=config.clip.dtype)
        processor = AutoProcessor.from_pretrained(config.clip.pretrained_model_name)
    else:
        clip_config_dict = {"projection_dim": config.clip.projection_dim, "logit_scale_init_value": config.clip.logit_scale_init_value, "logit_bias_init_value": config.clip.logit_bias_init_value, "dtype": config.clip.dtype}
        model = CLIPModel(text_config=text_config, vision_config=vision_config, **clip_config_dict)
        tokenizer = AutoTokenizer.from_pretrained(config.clip.pretrained_model_name)

    rng = jax.random.PRNGKey(config.seed)
    rng, rng_aug = jax.random.split(rng)
    
    # Potentially randomly sample a subset of the text for training
    max_length_words = config.data.max_length_words if config.data.augment_subsample_text else None

    # Rotation angles in rad
    rot_angles_90 = np.array([0.0, np.pi/2, np.pi, 3 * np.pi/2])

    # Initialize model if not using pre-trained; otherwise, use pre-trained weights
    if not config.clip.use_pretrained:

        # Pass a test batch through to initialize model
        images, captions = next(batches)
        input_ids, attention_mask = tokenize_captions(captions, tokenizer, config.text_config.max_length, max_length_words, rng_aug)
        batch = {"pixel_values": images, "input_ids": input_ids, "attention_mask": attention_mask}

        _, params = model.init_with_output(rng, batch["input_ids"][:1], batch["pixel_values"][:1], batch["attention_mask"][:1])
        logging.info("Loaded model for training from scratch")
    else:
        params = FrozenDict(model.params)
        logging.info(f"Loaded pretrained model {config.clip.pretrained_model_name}")        

    # Optionally, randomly initialize the vision and/or text models
    if (config.clip.random_init_vision or config.clip.random_init_text) and config.clip.use_pretrained:
        
        # Get randomly-initialized params
        params_init = model.module.init(rng, input_ids=np.zeros((1, model.config.text_config.max_length)), 
                        attention_mask=np.zeros((1, model.config.text_config.max_length)),
                        pixel_values=np.zeros((1, model.config.vision_config.image_size, model.config.vision_config.image_size, 3)),
                        position_ids=np.zeros((1, model.config.text_config.max_length)))
        
        if config.clip.random_init_vision:
            model.params['vision_model'] = params_init['params']['vision_model']
            model.params['visual_projection'] = params_init['params']['visual_projection']

            logging.info("Randomly initialized vision model")
        
        if config.clip.random_init_text:
            model.params['text_model'] = params_init['params']['text_model']
            model.params['text_projection'] = params_init['params']['text_projection']

            logging.info("Randomly initialized text model")

    logging.info(f"Number of parameters: {param_count(params)}")

    # Optionally convert type, for pretrained model
    if config.clip.dtype == "bfloat16" and config.clip.use_pretrained:
        model.params = model.to_bf16(model.params)
        logging.info("Converted model to bfloat16")

    ## Training config and loop
    
    best_fn = lambda metrics: metrics[f"val/{config.training.ckpt_best_metric}"]

    # At the top level
    mgr_options = orbax.checkpoint.CheckpointManagerOptions(create=True, step_prefix='step', max_to_keep=config.training.ckpt_keep_top_n, best_fn=best_fn, best_mode='max')
    ckpt_mgr = orbax.checkpoint.CheckpointManager(f"{workdir}/ckpts/", orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()), mgr_options)

    # Optimizer and schedule

    if config.optim.schedule == "constant":
        schedule = optax.linear_schedule(
            init_value=0.0,
            end_value=config.optim.learning_rate,
            transition_steps=config.training.warmup_steps,
        )
    elif config.optim.schedule == "cosine":
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.optim.learning_rate,
            warmup_steps=config.training.warmup_steps,
            decay_steps=config.training.n_train_steps,
        )
    else:
        raise ValueError(f"Invalid schedule: {config.optim.schedule}")
    
    tx = optax.adamw(learning_rate=schedule, weight_decay=config.optim.weight_decay)

    # State
    state = train_state.TrainState.create(apply_fn=model.__call__ if config.clip.use_pretrained else model.apply, params=params, tx=tx)
    pstate = replicate(state)

    # Log info about augmentations
    logging.info(f"Augment crop: {config.data.augment_crop}")
    logging.info(f"Augment rotate: {config.data.augment_rotate}")
    logging.info(f"Subsample text: {config.data.augment_subsample_text}. Max length: {max_length_words} words")

    if config.data.shuffle_within_batch:
        logging.info(f"Shuffling images within batch")

    logging.info("Starting training...")

    train_metrics = []
    with trange(config.training.n_train_steps) as steps:
        for step in steps:
            rng, rng_aug = jax.random.split(rng)
            images, captions = next(batches)
            images = np.array(images)

            # Augment images through random rotations
            if config.data.augment_rotate:
                rng_aug, _ = jax.random.split(rng_aug)
                rotation_angles = jax.random.choice(rng_aug, rot_angles_90, shape=(images.shape[0],))
                images = jax.vmap(partial(rotate, mode='constant', cval=1.))(images, rotation_angles)

            # Augment images through random crops
            # Otherwise, they'll be downsampled to the vision model's image size
            if config.data.augment_crop:
                rng_aug, _ = jax.random.split(rng_aug)
                images = jax.vmap(random_crop, in_axes=(None,0,None))(rng_aug, images, (model.config.vision_config.image_size, model.config.vision_config.image_size, 3))

            if config.clip.use_pretrained:
                # NOTE: Image arrays should be ints in the range [0, 255]
                captions = process_truncate_captions(captions, rng_aug, max_length_words=max_length_words)
                inputs = processor(text=captions, images=images * 255.,  return_tensors="np", padding="max_length", truncation=True, max_length=model.config.text_config.max_length)
                batch = inputs.data
            else:
                input_ids, attention_mask = tokenize_captions(captions, tokenizer, config.text_config.max_length, max_length_words, rng_aug)
                batch = {"pixel_values": images, "input_ids": input_ids, "attention_mask": attention_mask}
            
            # Optionally shuffle "pixel_values" within batch
            if config.data.shuffle_within_batch:
                batch["pixel_values"] = jax.random.permutation(rng, batch["pixel_values"], axis=0)

            # Split batch across devices
            batch = jax.tree_map(lambda x: np.split(x, num_local_devices, axis=0), batch)
            batch = jax.tree_map(lambda x: np.array(x, dtype=config.clip.dtype), batch)

            pstate, metrics = train_step(pstate, np.array(batch["input_ids"]), np.array(batch["pixel_values"]), np.array(batch["attention_mask"]), config.training.loss_type)
            steps.set_postfix(val=unreplicate(metrics["loss"]))
            train_metrics.append(metrics)

            # Log periodically
            if (step % config.training.log_every_steps == 0) and (step != 0) and (jax.process_index() == 0):
                train_metrics = common_utils.get_metrics(train_metrics)
                summary = {f"train/{k}": v for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()}

                writer.write_scalars(step, summary)
                train_metrics = []

                if config.wandb.log_train:
                    wandb.log({"train/step": step, **summary})

            rng_eval = jax.random.PRNGKey(config.seed)

            # Evaluate periodically
            # Evaluate before starting, hence no `and (step != 0)`
            if (step % config.training.eval_every_steps == 0) and (jax.process_index() == 0):

                # Log step at which evaluating 
                logging.info(f"Evaluating at step {step}")
                
                val_metrics = []
                val_batches = iter(val_ds)

                # Validate on 10 batches
                for _ in range(config.training.n_eval_batches):
                    images, captions = next(val_batches)
                    images = np.array(images)
                        
                    # Augment images through random rotations
                    if config.data.augment_rotate:
                        rotation_angles = jax.random.choice(rng_eval, rot_angles_90, shape=(images.shape[0],))  # Angles in radians
                        images = jax.vmap(partial(rotate, mode='constant', cval=1.))(images, rotation_angles)
                        
                    # Augment images through random crops
                    # Otherwise, they'll be downsampled to the vision model's image size
                    if config.data.augment_crop:
                        images = jax.vmap(random_crop, in_axes=(None,0,None))(rng_eval, images, (model.config.vision_config.image_size, model.config.vision_config.image_size, 3))

                    if config.clip.use_pretrained:
                        # NOTE: Image arrays should be ints in the range [0, 255]
                        captions = process_truncate_captions(captions, rng_eval, max_length_words=max_length_words)
                        inputs = processor(text=captions, images=images * 255.,  return_tensors="np", padding="max_length", truncation=True, max_length=model.config.text_config.max_length)
                        batch = inputs.data
                    else:
                        input_ids, attention_mask = tokenize_captions(captions, tokenizer, config.text_config.max_length, max_length_words, rng_eval)
                        batch = {"pixel_values": images, "input_ids": input_ids, "attention_mask": attention_mask}

                    # Split batch across devices
                    batch = jax.tree_map(lambda x: np.split(x, num_local_devices, axis=0), batch)
                    batch = jax.tree_map(lambda x: np.array(x, dtype=config.clip.dtype), batch)

                    metrics = eval_step(pstate, np.array(batch["input_ids"]), np.array(batch["pixel_values"]), np.array(batch["attention_mask"]), config.training.loss_type)
                    val_metrics.append(metrics)

                def serialize_metrics(metrics):
                    """Convert all values in the metrics dict to Python standard types."""
                    return {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in metrics.items()}

                val_metrics = common_utils.get_metrics(val_metrics)
                summary = {f"val/{k}": v for k, v in jax.tree_map(lambda x: x.mean(), val_metrics).items()}
                summary = serialize_metrics(summary)

                writer.write_scalars(step, summary)

                if config.wandb.log_train:
                    wandb.log({"val/step": step, **summary})

                # Save checkpoints periodically
                state_ckpt = unreplicate(pstate)
                save_args = orbax_utils.save_args_from_target(state_ckpt)
                ckpt_mgr.save(step, state_ckpt, save_kwargs={'save_args': save_args}, metrics=summary)

    logging.info("All done! Have a great day.")

    return unreplicate(pstate)


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the training or sampling hyperparameter configuration.",
        lock_config=True,
    )

    # Parse flags
    FLAGS(sys.argv)

    # Ensure TF does not see GPU and grab all GPU memory
    tf.config.experimental.set_visible_devices([], "GPU")

    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())
    logging.info("JAX total visible devices: %r", jax.device_count())

    # Start training run
    train(config=FLAGS.config, workdir=FLAGS.config.wandb.workdir)
