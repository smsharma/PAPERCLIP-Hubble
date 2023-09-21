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

from data.utils import make_dataloader, create_input_iter, get_abstracts_and_images

from tqdm import trange

import jax
import jax.numpy as np
import optax
import flax
from flax.core import FrozenDict
from flax.training import checkpoints, common_utils, train_state

import tensorflow as tf

from models.models import CLIPModel

replicate = flax.jax_utils.replicate
unreplicate = flax.jax_utils.unreplicate

logging.set_verbosity(logging.INFO)


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

    # Dump a yaml config file in the output directory
    with open(os.path.join(workdir, "config.yaml"), "w") as f:
        yaml.dump(config.to_dict(), f)

    writer = metric_writers.create_default_writer(logdir=workdir, just_logging=jax.process_index() != 0)

    abstracts, images, masks = get_abstracts_and_images(data_folder, abstracts_cycle_df)
    train_ds = make_dataloader(abstracts, masks, images, batch_size=32, seed=42)
    batches = create_input_iter(train_ds)

    logging.info("Loaded the %s dataset", config.data.dataset)

    ## Model configuration

    # Score and (optional) encoder model configs
    score_dict = FrozenDict(config.score)
    encoder_dict = FrozenDict(config.encoder)
    decoder_dict = FrozenDict(config.decoder)

    model = CLIPModel(...)

    rng = jax.random.PRNGKey(config.seed)
    rng, rng_params = jax.random.split(rng)

    # Pass a test batch through to initialize model
    x_batch, conditioning_batch, mask_batch = next(batches)
    _, params = model.init_with_output(
        {"sample": rng, "params": rng_params},
        x_batch[0],
        conditioning_batch[0],
        mask_batch[0],
    )

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

    state = train_state.TrainState.create(apply_fn=vdm.apply, params=params, tx=tx)
    pstate = replicate(state)

    logging.info("Starting training...")

    train_metrics = []
    with trange(config.training.n_train_steps) as steps:
        for step in steps:
            rng, *train_step_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
            train_step_rng = np.asarray(train_step_rng)
            x, conditioning, mask = next(batches)
            pstate, metrics = train_step(pstate, (x, conditioning, mask), train_step_rng, vdm, loss_vdm, config.training.unconditional_dropout, config.training.p_uncond)
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
