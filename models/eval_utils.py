import jax
import jax.numpy as np

def retrieval_eval_metric(outputs, k=[1, 5, 10]):
    """ Compute the top-k retrieval accuracy.
    """

    # Get outputs
    text_embeds = outputs["text_embeds"]
    image_embeds = outputs["image_embeds"]

    # Get shapes
    bs = text_embeds.shape[0]
    axis_size = jax.lax.psum(1, axis_name="batch")

    # Gather the embeddings from all devices
    all_text_embeds = jax.lax.all_gather(text_embeds, axis_name="batch").reshape(-1, text_embeds.shape[-1])
    all_image_embeds = jax.lax.all_gather(image_embeds, axis_name="batch").reshape(-1, image_embeds.shape[-1])

    # Compute the full matrix of logitseval
    all_logits = np.matmul(all_text_embeds, all_image_embeds.T)

    # Compute the global top-k indices for the maximum k value
    max_k = max(k)
    top_k_indices = np.argsort(all_logits, axis=-1)[:, -max_k:]

    # Compute the correct indices for each row
    correct_indices = np.arange(bs * axis_size)[:, None]

    metrics = {}
    for current_k in k:
        # Check if the correct image (diagonal) is in the current top-k for each text embedding
        correct_in_top_k = np.any(top_k_indices[:, -current_k:] == correct_indices, axis=-1)
        accuracy = np.mean(correct_in_top_k.astype(np.float32))
        metrics[f"top_{current_k}_accuracy"] = accuracy

    return metrics