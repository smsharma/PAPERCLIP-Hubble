import jax
import jax.numpy as np
import flax
from ml_collections import ConfigDict
import jax.numpy as jnp

import sys

sys.path.append("../")
from models.losses import sigmoid_loss

from functools import partial


@jax.jit
def train_step(state, batch):
    """Train for a single step."""

    input_ids, images, attention_mask = batch

    def loss_fn(params):
        outputs = state.apply_fn(params, input_ids, images, attention_mask)
        loss = sigmoid_loss(outputs)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    metrics = {"loss": jnp.mean(loss)}
    return new_state, metrics


def param_count(pytree):
    """Count the number of parameters in a pytree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(pytree))


def to_wandb_config(d: ConfigDict, parent_key: str = "", sep: str = "."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, ConfigDict):
            items.extend(to_wandb_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
