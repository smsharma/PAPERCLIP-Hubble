import jax
import jax.numpy as np

# Enable float64 for numerical stability
jax.config.update("jax_enable_x64", True)

def mini_batch_sigmoid_loss(text_embeds, image_embeds, logit_scale, logit_bias, negative_samples):
    """Positive samples are on the diagonal"""
    bs = text_embeds.shape[0]
    if negative_samples:  # All negative samples 
        labels = -np.ones((bs, bs))
    else:  # Positive samples are on the diagonal
        labels = 2 * np.eye(bs) - np.ones((bs, bs))
    logits = np.matmul(text_embeds, image_embeds.T) * logit_scale + logit_bias
    
    # Increase precision for large batches
    logits = logits.astype(np.float64)
    return -np.mean(jax.nn.log_sigmoid(labels * logits))

def sigmoid_loss(outputs):
    """ SigLIP loss (https://arxiv.org/abs/2303.15343); 
        Adapted from https://github.com/borisdayma/clip-jax/blob/main/training/train.py
    """

    # Get outputs
    ztxt = outputs["text_embeds"]
    zimg = outputs["image_embeds"]
    logit_scale = outputs["logit_scale"]  # Same as temperature
    logit_bias = outputs.get("logit_bias", 0.)

    # Number of chunks (devices)
    axis_size = jax.lax.psum(1, axis_name="batch")

    # Calculate local device loss
    loss = mini_batch_sigmoid_loss(ztxt, zimg, logit_scale, logit_bias, negative_samples=False)

    # Add negative losses
    def add_negative_loss(i, carrys):
        cumul_loss, zimg = carrys
        
        # Shift image_embeds
        zimg = jax.lax.ppermute(
            zimg, axis_name="batch", perm=[(j, (j - 1) % axis_size) for j in range(axis_size)]
        )
        # Add loss (all negative samples)
        cumul_loss += mini_batch_sigmoid_loss(
            ztxt, zimg, logit_scale, logit_bias, negative_samples=True
        )
        
        return cumul_loss, zimg

    loss, _ = jax.lax.fori_loop(0, axis_size - 1, add_negative_loss, (loss, zimg))
    loss = loss / axis_size

    loss = loss.reshape((-1,))

    # Average loss across devices
    loss = np.mean(loss)
    return loss

def all_gather(z, roll=False, only_others=False):
    """All gather and flatten first two dims."""

    def gather_flat(x):
        x = jax.lax.all_gather(x, "batch")
        if roll or only_others:
            # Each device moves "its" chunk to the beginning. Simplies loss/acc calcs.
            x = np.roll(x, -jax.lax.axis_index("batch"), axis=0)
            if only_others:
                x = x[1:]
        return np.concatenate(x, 0)  # Fold in "device" and "batch" dims.

    return jax.tree_map(gather_flat, z)


def softmax_loss(outputs):
    """Softmax loss following the CLIP paper. Factorized to reduce memory cost.
        Adapted from `big_vision` repo https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/contrastive.py
    """

    # Get outputs
    zimg = outputs["image_embeds"]
    ztxt = outputs["text_embeds"]
    logit_scale = outputs["logit_scale"]  # Same as temperature
    logit_bias = outputs.get("logit_bias", 0.)

    def unidirectional_loss(z1, z2, logit_scale, logit_bias):
        """ Compute the unidirectional z1 -> z2 contrastive loss between two sets of embeddings.
        """        
        z2 = all_gather(z2, roll=True)
        logits = np.dot(z1, z2.T) * logit_scale + logit_bias

        # Softmax across the larger gathered axis, taking advantage of the
        # fact that positives are known to be on the diagonal.
        loss = -(np.diag(logits) - jax.scipy.special.logsumexp(logits, axis=-1))
        return loss.mean()

    loss = 0.
    for row, col in [(zimg, ztxt), (ztxt, zimg)]:
        loss_dir = unidirectional_loss(row, col, logit_scale, logit_bias)
        loss += 0.5 * loss_dir

    loss = jax.lax.pmean(loss, "batch")
    return loss