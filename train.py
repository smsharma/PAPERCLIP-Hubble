import sys
import os
import yaml
from functools import partial

sys.path.append("./")
sys.path.append("../")

from absl import flags, logging
from ml_collections import config_flags, ConfigDict
from clu import metric_writers
import wandb

import pandas as pd

import jax
import jax.numpy as np
import optax
import flax
from flax.core import FrozenDict
from flax.training import common_utils, train_state, orbax_utils
from flax import traverse_util
import orbax.checkpoint as ocp
from dm_pix import rotate, random_crop, random_flip_up_down, random_flip_left_right
import tensorflow as tf
from tqdm import trange

from transformers import AutoProcessor, FlaxCLIPModel

from utils.dataset_utils import make_dataloader
from utils.text_utils import process_truncate_captions
from models.train_utils import param_count, train_step, eval_step, to_wandb_config
from models.clip_transfer import FlaxCLIPModelTransfer

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
        wandb.define_metric(
            "*", step_metric="train/step"
        )  # Set default x-axis as 'train/step'

        # If loading ckpt, set the dir for that run
        if config.training.load_ckpt:
            workdir_load_ckpt = os.path.join(workdir, run.group, config.training.ckpt_run_name)

        workdir = os.path.join(workdir, run.group, run.name)


        # Recursively create workdir
        os.makedirs(workdir, exist_ok=True)

    # Devices
    num_local_devices = jax.local_device_count()
    num_hosts = jax.process_count()

    # Dump a yaml config file in the output directory
    with open(os.path.join(workdir, "config.yaml"), "w") as f:
        yaml.dump(config.to_dict(), f)

    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.process_index() != 0
    )

    # Set up data

    # Find all TFRecord files and make datasets
    files_train = tf.io.gfile.glob(
        f"{config.data.data_dir}/{config.data.tfrecords_dir}/*train*.tfrecord"
    )
    train_ds = make_dataloader(
        files_train,
        batch_size=config.training.batch_size,
        seed=config.seed,
        split="train",
        caption_type=config.data.caption_type,
        shuffle=True,
    )

    files_val = tf.io.gfile.glob(
        f"{config.data.data_dir}/{config.data.tfrecords_dir}/*val*.tfrecord"
    )
    val_ds = make_dataloader(
        files_val,
        batch_size=config.training.batch_size_val,
        seed=config.seed,
        split="val",
        caption_type=config.data.caption_type,
        shuffle=True,
    )
    
    batches = iter(train_ds)

    # Convert to jnp type
    dtype = getattr(np, config.clip.dtype)

    logging.info("Loaded the dataset")
    logging.info(f"Using caption type: {config.data.caption_type}")

    # # Model configs
    # # NOTE: From legacy code; not used currently, but potentially useful if options are fed to create custom CLIP model
    # text_config = FrozenDict(config.text_config)
    # vision_config = FrozenDict(config.vision_config)
    # clip_config = FrozenDict(config.clip)

    # Use pre-trained model or train from scratch
    if config.clip.use_pretrained:
        model = FlaxCLIPModel.from_pretrained(
            config.clip.pretrained_model_name, dtype=dtype
        )
        processor = AutoProcessor.from_pretrained(config.clip.pretrained_model_name)
    else:
        raise NotImplementedError

    rng = jax.random.PRNGKey(config.seed)
    rng, rng_aug = jax.random.split(rng)

    # Potentially randomly sample a subset of the text for training
    max_length_words = (
        config.data.max_length_words if config.data.augment_subsample_text else None
    )

    # Rotation angles in rad
    rot_angles_90 = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])

    # Pre-trained model or train from scratch
    if config.clip.use_pretrained:
        # Optionally, randomly initialize the vision and/or text models
        if config.clip.random_init_vision or config.clip.random_init_text:

            # Get randomly-initialized params
            params_init = model.module.init(
                rng,
                input_ids=np.zeros((1, model.config.text_config.max_position_embeddings)),
                attention_mask=np.zeros((1, model.config.text_config.max_position_embeddings)),
                pixel_values=np.zeros(
                    (
                        1,
                        model.config.vision_config.image_size,
                        model.config.vision_config.image_size,
                        3,
                    )
                ),
                position_ids=np.zeros((1, model.config.text_config.max_position_embeddings)),
            )["params"]

            if config.clip.random_init_vision:
                model.params["vision_model"] = params_init["vision_model"]
                model.params["visual_projection"] = params_init["visual_projection"]

                logging.info("Randomly initialized vision model")

            if config.clip.random_init_text:
                model.params["text_model"] = params_init["text_model"]
                model.params["text_projection"] = params_init["text_projection"]

                logging.info("Randomly initialized text model")

        logging.info(f"Loaded pretrained model {config.clip.pretrained_model_name}")

    else:
        raise NotImplementedError

    if config.clip.transfer_head:

        model_transfer = FlaxCLIPModelTransfer(
            config=model.config, dtype=dtype, d_head=config.clip.d_transfer_head
        )

        # Transfer text and vision backbones
        model_transfer.params["text_model"]["text_backbone"] = model.params[
            "text_model"
        ]
        model_transfer.params["vision_model"]["vision_backbone"] = model.params[
            "vision_model"
        ]

        # Complete transfer
        model = model_transfer
        logging.info(
            f"Transferred pretrained model {config.clip.pretrained_model_name} with {config.clip.d_transfer_head}-dim head"
        )

    # Optionally convert type
    if config.clip.dtype == "bfloat16":
        model.params = model.to_bf16(model.params)
        logging.info("Converted model to bfloat16")

    logging.info(f"Number of parameters: {param_count(model.params)}")
    params = model.params  # FrozenDict(model.params)

    ## Training config and loop

    # Checkpoint manager

    best_fn = lambda metrics: metrics[f"val/{config.training.ckpt_best_metric}"]
    mgr_options = ocp.CheckpointManagerOptions(
        step_prefix=f"step",
        max_to_keep=config.training.ckpt_keep_top_n,
        best_fn=best_fn,
        best_mode=config.training.ckpt_best_metric_best_mode,
    )

    ckpt_mgr = ocp.CheckpointManager(
        ocp.test_utils.create_empty(f"{workdir}/ckpts/"),
        options=mgr_options,
    )

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

    if config.clip.transfer_head:
        # Partition optimizer into trainable and frozen
        partition_optimizers = {
            "trainable": optax.adamw(
                learning_rate=schedule, weight_decay=config.optim.weight_decay
            ),
            "frozen": optax.set_to_zero(),
        }
        param_partitions = traverse_util.path_aware_map(
            lambda path, v: (
                "frozen"
                if (("vision_backbone" in path) or ("text_backbone" in path))
                else "trainable"
            ),
            params,
        )

        tx = optax.multi_transform(partition_optimizers, param_partitions)
    else:
        tx = optax.adamw(learning_rate=schedule, weight_decay=config.optim.weight_decay)

    # State
    state = train_state.TrainState.create(
        apply_fn=model.__call__ if (config.clip.use_pretrained) else model.apply,
        params=params,
        tx=tx,
    )

    # Load checkpoint
    if config.training.load_ckpt:

        ckpt_mgr_load_ckpt = ocp.CheckpointManager(
            f"{workdir_load_ckpt}/ckpts/",
            options=mgr_options,
        )

        logging.info(f"Loading checkpoint from run {config.training.ckpt_run_name}")

        restored_state = ckpt_mgr_load_ckpt.restore(
            ckpt_mgr_load_ckpt.latest_step(),
            args=ocp.args.StandardRestore(state)
        )

        if state is restored_state:
            raise FileNotFoundError(f"Did not load checkpoint correctly")
        else:
            state = restored_state
            logging.info(f"Loaded checkpoint from step {ckpt_mgr_load_ckpt.latest_step()}")

    pstate = replicate(state)

    # Log info about augmentations
    logging.info(f"Augment crop: {config.data.augment_crop}")
    logging.info(
        f"Augment rotate: {config.data.augment_rotate}, {config.data.augment_rotate_type}"
    )
    logging.info(
        f"Subsample text: {config.data.augment_subsample_text}. Max length: {max_length_words} words"
    )

    if config.data.shuffle_within_batch:
        logging.info(f"Shuffling images within batch")

    # If matching to sum1 summary
    if config.sum1.use_sum1:
        logging.info(
            f"Matching to summary {config.sum1.summaries_filename} with sum1 {config.sum1.sum1_filename}"
        )
        # Combine with data dir and add .csv extension
        summaries_filename = os.path.join(
            config.data.data_dir, f"{config.sum1.summaries_filename}.csv"
        )
        sum1_filename = os.path.join(
            config.data.data_dir, f"{config.sum1.sum1_filename}.csv"
        )

        # Open dataframes
        df_summaries = pd.read_csv(summaries_filename)
        df_sum1 = pd.read_csv(sum1_filename)

        # Merge on proposal ID
        df_sum_merged = pd.merge(df_summaries, df_sum1, on="proposal_id")
    else:
        df_sum_merged = None

    logging.info("Starting training...")

    train_metrics = []
    with trange(config.training.n_train_steps) as steps:
        for step in steps:

            # Eval portion

            # Evaluate before starting, hence no `and (step != 0)`
            if (step % config.training.eval_every_steps == 0) and (
                jax.process_index() == 0
            ):

                # Use same rng for eval every time
                rng_eval = jax.random.PRNGKey(config.seed)

                # Log step at which evaluating
                logging.info(f"Evaluating at step {step}")

                val_metrics = []
                val_batches = iter(val_ds)

                total_batches = sum(1 for _ in val_ds) - 1
                current_batch = 0

                # Validate on 10 batches
                for images, captions in val_batches:

                    # Break if we've reached the end of the dataset (except for the last batch, which is likely partial)
                    if current_batch == total_batches - 1:
                        break

                    images = np.array(images)

                    # NOTE: Don't augment images for eval! NOTE: Changed my mind.
                    # Augment images through random rotations and flips
                    if config.data.augment_rotate:

                        # Rotations
                        rng_eval, _ = jax.random.split(rng_eval)
                        if config.data.augment_rotate_type == "continuous":
                            rotation_angles = (
                                jax.random.uniform(rng_eval, shape=(images.shape[0],))
                                * 2
                                * np.pi
                            )  # Angles in radians
                        elif config.data.augment_rotate_type == "discrete":
                            rotation_angles = jax.random.choice(
                                rng_eval, rot_angles_90, shape=(images.shape[0],)
                            )  # Angles in radians
                        else:
                            raise ValueError(
                                f"Invalid augment_rotate_type: {config.data.augment_rotate_type}"
                            )
                        images = jax.vmap(partial(rotate, mode="constant", cval=1.0))(
                            images, rotation_angles
                        )

                        # Flips
                        rng_eval, _ = jax.random.split(rng_eval)
                        images = jax.vmap(partial(random_flip_up_down, key=rng_eval))(
                            image=images
                        )

                        rng_eval, _ = jax.random.split(rng_eval)
                        images = jax.vmap(
                            partial(random_flip_left_right, key=rng_eval)
                        )(image=images)

                    # Augment images through random crops
                    # Otherwise, they'll be downsampled to the vision model's image size
                    if config.data.augment_crop:
                        images = jax.vmap(random_crop, in_axes=(None, 0, None))(
                            rng_eval,
                            images,
                            (
                                model.config.vision_config.image_size,
                                model.config.vision_config.image_size,
                                3,
                            ),
                        )

                    # NOTE: Image arrays should be ints in the range [0, 255] here
                    captions = process_truncate_captions(
                        captions,
                        rng_eval,
                        max_length_words=max_length_words,
                        use_sum1=config.sum1.use_sum1,
                        df_sum_merged=df_sum_merged,
                    )
                    inputs = processor(
                        text=captions,
                        images=(images * 255.0).astype(np.uint8),
                        return_tensors="np",
                        padding="max_length",
                        truncation=True,
                        max_length=model.config.text_config.max_position_embeddings,
                    )
                    batch = inputs.data

                    # Split batch across devices
                    batch = jax.tree_map(
                        lambda x: np.split(x, num_local_devices, axis=0), batch
                    )
                    batch = jax.tree_map(lambda x: np.array(x, dtype=dtype), batch)

                    metrics = eval_step(
                        pstate,
                        np.array(batch["input_ids"]),
                        np.array(batch["pixel_values"]),
                        np.array(batch["attention_mask"]),
                        config.training.loss_type,
                    )
                    val_metrics.append(metrics)

                    current_batch += 1  # Increment batch counter

                def serialize_metrics(metrics):
                    """Convert all values in the metrics dict to Python standard types."""
                    return {
                        k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                        for k, v in metrics.items()
                    }

                val_metrics = common_utils.get_metrics(val_metrics)
                summary = {
                    f"val/{k}": v
                    for k, v in jax.tree_map(lambda x: x.mean(), val_metrics).items()
                }
                summary = serialize_metrics(summary)

                writer.write_scalars(step, summary)

                if config.wandb.log_train:
                    wandb.log({"val/step": step, **summary})

                # Save checkpoints periodically
                state_ckpt = unreplicate(pstate)

                ckpt_mgr.save(
                    step,
                    args=ocp.args.StandardSave(state_ckpt),
                    metrics=summary
                )
                ckpt_mgr.wait_until_finished()

            # Train portion

            rng, rng_aug = jax.random.split(rng)
            images, captions = next(batches)
            images = np.array(images)

            # Augment images through random rotations and flips
            if config.data.augment_rotate:

                # Rotations
                rng_aug, _ = jax.random.split(rng_aug)
                if config.data.augment_rotate_type == "continuous":
                    rotation_angles = (
                        jax.random.uniform(rng_aug, shape=(images.shape[0],))
                        * 2
                        * np.pi
                    )  # Angles in radians
                elif config.data.augment_rotate_type == "discrete":
                    rotation_angles = jax.random.choice(
                        rng_aug, rot_angles_90, shape=(images.shape[0],)
                    )  # Angles in radians
                else:
                    raise ValueError(
                        f"Invalid augment_rotate_type: {config.data.augment_rotate_type}"
                    )
                images = jax.vmap(partial(rotate, mode="constant", cval=1.0))(
                    images, rotation_angles
                )

                # Flips
                rng_aug, _ = jax.random.split(rng_aug)
                images = jax.vmap(partial(random_flip_up_down, key=rng_aug))(
                    image=images
                )

                rng_aug, _ = jax.random.split(rng_aug)
                images = jax.vmap(partial(random_flip_left_right, key=rng_aug))(
                    image=images
                )

            # Augment images through random crops
            # Otherwise, they'll be downsampled to the vision model's image size
            if config.data.augment_crop:
                rng_aug, _ = jax.random.split(rng_aug)
                images = jax.vmap(random_crop, in_axes=(None, 0, None))(
                    rng_aug,
                    images,
                    (
                        model.config.vision_config.image_size,
                        model.config.vision_config.image_size,
                        3,
                    ),
                )

            # NOTE: Image arrays should be ints in the range [0, 255] here
            captions = process_truncate_captions(
                captions,
                rng_aug,
                max_length_words=max_length_words,
                use_sum1=config.sum1.use_sum1,
                df_sum_merged=df_sum_merged,
            )
            inputs = processor(
                text=captions,
                images=(images * 255.0).astype(np.uint8),
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=model.config.text_config.max_position_embeddings,
            )
            batch = inputs.data

            # Optionally shuffle "pixel_values" within batch
            if config.data.shuffle_within_batch:
                batch["pixel_values"] = jax.random.permutation(
                    rng, batch["pixel_values"], axis=0
                )

            # Split batch across devices
            batch = jax.tree_map(
                lambda x: np.split(x, num_local_devices, axis=0), batch
            )
            batch = jax.tree_map(lambda x: np.array(x, dtype=dtype), batch)

            pstate, metrics = train_step(
                pstate,
                np.array(batch["input_ids"]),
                np.array(batch["pixel_values"]),
                np.array(batch["attention_mask"]),
                config.training.loss_type,
            )
            steps.set_postfix(val=unreplicate(metrics["loss"]))
            train_metrics.append(metrics)

            # Log periodically
            if (
                (step % config.training.log_every_steps == 0)
                and (step != 0)
                and (jax.process_index() == 0)
            ):
                train_metrics = common_utils.get_metrics(train_metrics)
                summary = {
                    f"train/{k}": v
                    for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()
                }

                writer.write_scalars(step, summary)
                train_metrics = []

                if config.wandb.log_train:
                    wandb.log({"train/step": step, **summary})

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
