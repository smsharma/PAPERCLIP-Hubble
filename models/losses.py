import jax
import jax.numpy as np

# Enable float64 for numerical stability
jax.config.update("jax_enable_x64", True)

def mini_batch_sigmoid_loss(text_embeds, image_embeds, logit_scale, logit_bias, negative_samples):

    bs = text_embeds.shape[0]
    
    if negative_samples:
        labels = -np.ones((bs, bs))
    else:
        labels = 2 * np.eye(bs) - np.ones((bs, bs))
        
    logits = np.matmul(text_embeds, image_embeds.T) * logit_scale + logit_bias
    
    # Increase numerical stability
    logits = logits.astype(np.float64)

    # Use log_sigmoid for numerical stability
    loss = -np.mean(jax.nn.log_sigmoid(labels * logits)) 
    
    return loss

def sigmoid_loss(outputs):

    text_embeds = outputs["text_embeds"]
    image_embeds = outputs["image_embeds"]
    logit_scale = outputs["logit_scale"]
    logit_bias = outputs["logit_bias"]

    bs = text_embeds.shape[0]

    # Compute positive loss
    loss = mini_batch_sigmoid_loss(text_embeds, image_embeds, logit_scale, logit_bias, False)

    # Negative pairs
    shifted_image_embeds = [np.roll(image_embeds, -i, axis=0) for i in range(1, bs)]
    shifted_image_embeds = np.stack(shifted_image_embeds)
    
    all_neg_logits = np.einsum("bi,aji->abj", text_embeds, shifted_image_embeds)
    all_neg_logits = all_neg_logits * logit_scale + logit_bias
    all_neg_logits = all_neg_logits.astype(np.float64)

    neg_labels = -np.ones(all_neg_logits.shape)
    neg_loss = -np.mean(jax.nn.log_sigmoid(neg_labels * all_neg_logits))

    loss += (bs - 1) * neg_loss
    loss /= bs

    return loss
