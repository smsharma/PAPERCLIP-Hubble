import jax.numpy as np


def mini_batch_sigmoid_loss(text_embeds, image_embeds, logit_scale, logit_bias, negative_samples):
    """Positive samples are on the diagonal"""
    bs = text_embeds.shape[0]
    if negative_samples:
        labels = -np.ones((bs, bs))
    else:
        labels = 2 * np.eye(bs) - np.ones((bs, bs))
    logits = np.matmul(text_embeds, image_embeds.T) * logit_scale + logit_bias
    return -np.mean(np.log(1 + np.exp(-labels * logits)))


def sigmoid_loss(outputs):
    text_embeds = outputs["text_embeds"]
    image_embeds = outputs["image_embeds"]
    logit_scale = outputs["logit_scale"]
    logit_bias = outputs["logit_bias"]

    bs = text_embeds.shape[0]

    # Compute the positive samples loss
    loss = mini_batch_sigmoid_loss(text_embeds, image_embeds, logit_scale, logit_bias, negative_samples=False)

    # Create a tensor of all shifted versions of image embeddings
    shifted_image_embeds = np.stack([np.roll(image_embeds, shift=-i, axis=0) for i in range(1, bs)])

    # Compute the negative samples logits using einsum
    all_neg_logits = np.einsum("bi,aji->abj", text_embeds, shifted_image_embeds)
    all_neg_logits = all_neg_logits * logit_scale + logit_bias

    neg_labels = -np.ones(all_neg_logits.shape)
    neg_loss = -np.mean(np.log(1 + np.exp(-neg_labels * all_neg_logits)))

    loss = (loss + (bs - 1) * neg_loss) / bs

    return loss
