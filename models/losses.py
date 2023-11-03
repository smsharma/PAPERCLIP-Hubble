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
    text_embeds = outputs["text_embeds"]
    image_embeds = outputs["image_embeds"]
    logit_scale = outputs["logit_scale"]
    logit_bias = outputs.get("logit_bias", 0.)

    # Number of chunks (devices)
    axis_size = jax.lax.psum(1, axis_name="batch")

    # Calculate local device loss
    loss = mini_batch_sigmoid_loss(text_embeds, image_embeds, logit_scale, logit_bias, negative_samples=False)

    # Add negative losses
    def add_negative_loss(i, carrys):
        cumul_loss, image_embeds = carrys
        
        # Shift image_embeds
        image_embeds = jax.lax.ppermute(
            image_embeds, axis_name="batch", perm=[(j, (j - 1) % axis_size) for j in range(axis_size)]
        )
        # Add loss (all negative samples)
        cumul_loss += mini_batch_sigmoid_loss(
            text_embeds, image_embeds, logit_scale, logit_bias, negative_samples=True
        )
        
        return cumul_loss, image_embeds

    loss, _ = jax.lax.fori_loop(0, axis_size - 1, add_negative_loss, (loss, image_embeds))
    loss = loss / axis_size

    loss = loss.reshape((-1,))

    # Average loss across devices
    loss = np.mean(loss)
    return loss