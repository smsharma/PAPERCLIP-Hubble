import jax
import jax.numpy as np
import flax
from ml_collections import ConfigDict

import sys

sys.path.append("../")
from models.losses import sigmoid_loss
from models.eval import retrieval_eval_metric

from functools import partial


@partial(jax.pmap, axis_name="batch")
def train_step(state, input_ids, images, attention_mask):
    """Train for a single step."""

    def loss_fn(params):
        outputs = state.apply_fn(params, input_ids, images, attention_mask)
        loss = sigmoid_loss(outputs)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.lax.pmean(grads, "batch")
    new_state = state.apply_gradients(grads=grads)
    metrics = {"loss": jax.lax.pmean(loss, "batch")}
    return new_state, metrics

@partial(jax.pmap, axis_name="batch")
def eval_step(state, input_ids, images, attention_mask):
    """Eval step."""

    def loss_fn(params):
        outputs = state.apply_fn(params, input_ids, images, attention_mask)
        loss = sigmoid_loss(outputs)
        retrieval_metrics = retrieval_eval_metric(outputs)
        return loss, retrieval_metrics

    loss, retrieval_metrics = loss_fn(state.params)
    
    metrics = {"loss": jax.lax.pmean(loss, "batch")}
    for key, value in retrieval_metrics.items():
        metrics[key] = jax.lax.pmean(value, "batch")

    return metrics
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
