import sys
import os
import yaml

from absl import flags, logging
from absl import logging
import ml_collections
from ml_collections import config_flags
from clu import metric_writers
import wandb

sys.path.append("./")
sys.path.append("../")

from tqdm import trange

import jax
import jax.numpy as np
import optax
import flax
from flax.core import FrozenDict
from flax.training import checkpoints, common_utils, train_state

from transformers import AutoTokenizer

import tensorflow as tf

import dataclasses

from models.clip import CLIPModel
from models.dataset_utils import make_dataloader, create_input_iter
from models.train_utils import param_count, train_step, to_wandb_config

replicate = flax.jax_utils.replicate
unreplicate = flax.jax_utils.unreplicate

logging.set_verbosity(logging.INFO)


def tokenize_captions(captions, tokenizer):
    captions = captions.numpy().tolist()
    captions = [c.decode("utf-8") for c in captions]
    captions = tokenizer(captions, padding="max_length", truncation=True, max_length=300, return_tensors="np")
    return captions["input_ids"], captions["attention_mask"]


def train(config: ml_collections.ConfigDict, workdir: str = "./logging/") -> train_state.TrainState:
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
    files = ["./data/observations.tfrecord"]
    ds = make_dataloader(files, batch_size=config.training.batch_size, seed=config.seed)
    # batches = create_input_iter(ds)
    batches = iter(ds)

    logging.info("Loaded the dataset")

    ## Model configuration

    # Score and (optional) encoder model configs
    text_config = FrozenDict(config.text_config)
    vision_config = FrozenDict(config.vision_config)
    clip_config = FrozenDict(config.clip)

    model = CLIPModel(text_config=text_config, vision_config=vision_config, **clip_config)

    rng = jax.random.PRNGKey(config.seed)
    rng, _ = jax.random.split(rng)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Pass a test batch through to initialize model
    images, captions = next(batches)
    input_ids, attention_mask = tokenize_captions(captions, tokenizer)

    batch = {"pixel_values": images, "input_ids": input_ids, "attention_mask": attention_mask}

    # Just pass one element of the batch to initialize the model

    _, params = model.init_with_output(rng, batch["input_ids"][:1], batch["pixel_values"][:1], batch["attention_mask"][:1])

    logging.info("Instantiated the model")
    logging.info("Number of parameters: %d", param_count(params))

    ## Training config and loop

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.optim.learning_rate,
        warmup_steps=config.training.warmup_steps,
        decay_steps=config.training.n_train_steps,
    )
    tx = optax.adamw(learning_rate=schedule, weight_decay=config.optim.weight_decay)

    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    pstate = replicate(state)

    logging.info("Starting training...")

    train_metrics = []
    with trange(config.training.n_train_steps) as steps:
        for step in steps:
            rng, _ = jax.random.split(rng)
            images, captions = next(batches)
            input_ids, attention_mask = tokenize_captions(captions, tokenizer)

            batch = {"pixel_values": images, "input_ids": input_ids, "attention_mask": attention_mask}
            batch = jax.tree_map(lambda x: np.array(x, dtype=config.vision_config.dtype), batch)
            batch = jax.tree_map(lambda x: np.split(x, num_local_devices, axis=0), batch)

            pstate, metrics = train_step(pstate, np.array(batch["input_ids"]), np.array(batch["pixel_values"]), np.array(batch["attention_mask"]))
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

            # Save checkpoints periodically
            if (step % config.training.save_every_steps == 0) and (step != 0) and (jax.process_index() == 0):
                state_ckpt = unreplicate(pstate)
                checkpoints.save_checkpoint(
                    ckpt_dir=workdir,
                    target=state_ckpt,
                    step=step,
                    overwrite=True,
                    keep=np.inf,
                )

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
