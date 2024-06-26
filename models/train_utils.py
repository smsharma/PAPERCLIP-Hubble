import jax
import jax.numpy as np
import flax
from ml_collections import ConfigDict

import sys

sys.path.append("../")
from models.losses import sigmoid_loss, softmax_loss
from models.eval_utils import retrieval_eval_metric

from functools import partial


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(4,))
def train_step(state, input_ids, images, attention_mask, loss_type="sigmoid"):
    """Train for a single step."""

    def loss_fn(params):
        outputs = state.apply_fn(input_ids=input_ids, pixel_values=images, attention_mask=attention_mask, params=params)
        outputs['logit_scale'] = params['logit_scale']
        outputs['logit_bias'] = params.get('logit_bias', 0.)

        if loss_type == "sigmoid":
            loss = sigmoid_loss(outputs)
        elif loss_type == "softmax":
            loss = softmax_loss(outputs)
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")
        
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.lax.pmean(grads, "batch")
    new_state = state.apply_gradients(grads=grads)
    metrics = {"loss": jax.lax.pmean(loss, "batch")}
    return new_state, metrics

@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(4,))
def eval_step(state, input_ids, images, attention_mask, loss_type="sigmoid"):
    """Eval step."""

    def loss_fn(params):
        outputs = state.apply_fn(input_ids=input_ids, pixel_values=images, attention_mask=attention_mask, params=params)
        outputs['logit_scale'] = params['logit_scale']
        outputs['logit_bias'] = params.get('logit_bias', 0.)

        if loss_type == "sigmoid":
            loss = sigmoid_loss(outputs)
        elif loss_type == "softmax":
            loss = softmax_loss(outputs)
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")

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
