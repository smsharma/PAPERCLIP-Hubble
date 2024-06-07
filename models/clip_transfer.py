from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from transformers.models.clip import FlaxCLIPPreTrainedModel, CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from transformers.models.clip.modeling_flax_clip import FlaxCLIPTextTransformer, FlaxCLIPVisionTransformer, FlaxCLIPOutput

class TextModelTransfer(nn.Module):
    text_config: CLIPTextConfig
    dtype: jnp.dtype = jnp.float32
    h1: int = 768
    h2: int = 512

    @nn.compact
    def __call__(self, 
            input_ids,
            attention_mask,
            position_ids,
            deterministic: bool = True,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
         ):
        
        out = FlaxCLIPTextTransformer(self.text_config, dtype=self.dtype, name="text_backbone")(input_ids, attention_mask, position_ids, deterministic, output_attentions, output_hidden_states, return_dict)
        emb = nn.Dense(self.h1, dtype=self.dtype)(out[1])
        emb = nn.gelu(emb)
        emb = nn.Dense(self.h2, dtype=self.dtype)(emb)
        out['pooler_output'] = emb
        return out
    
class VisionModelTransfer(nn.Module):
    vision_config: CLIPVisionConfig
    dtype: jnp.dtype = jnp.float32
    h1: int = 768
    h2: int = 512

    @nn.compact
    def __call__(self,
            pixel_values=None,
            deterministic: bool = True,
            output_attentions=None,
            output_hidden_states=None,
            return_dict: bool = True,
         ):
        
        out = FlaxCLIPVisionTransformer(self.vision_config, dtype=self.dtype, name="vision_backbone")(pixel_values, deterministic, output_attentions, output_hidden_states, return_dict)
        emb = nn.Dense(self.h1, dtype=self.dtype)(out[1])
        emb = nn.gelu(emb)
        emb = nn.Dense(self.h2, dtype=self.dtype)(emb)
        out['pooler_output'] = emb
        return out

class FlaxCLIPModuleTransfer(nn.Module):
    config: CLIPConfig
    dtype: jnp.dtype = jnp.float32
    d_head: int = 1024

    def setup(self):
        text_config = self.config.text_config
        vision_config = self.config.vision_config

        self.projection_dim = self.config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = TextModelTransfer(text_config, dtype=self.dtype, h1=self.d_head, h2=text_config.hidden_size)
        self.vision_model = VisionModelTransfer(vision_config, dtype=self.dtype, h1=self.d_head, h2=vision_config.hidden_size)

        self.visual_projection = nn.Dense(
            self.projection_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
            use_bias=False,
        )
        self.text_projection = nn.Dense(
            self.projection_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
            use_bias=False,
        )

        self.logit_scale = self.param(
            "logit_scale", lambda _, shape: jnp.ones(shape) * self.config.logit_scale_init_value, []
        )

    def __call__(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / jnp.linalg.norm(image_embeds, axis=-1, keepdims=True)
        text_embeds = text_embeds / jnp.linalg.norm(text_embeds, axis=-1, keepdims=True)

        # cosine similarity as logits
        logit_scale = jnp.exp(self.logit_scale)
        logits_per_text = jnp.matmul(text_embeds, image_embeds.T) * logit_scale
        logits_per_image = logits_per_text.T

        if not return_dict:
            return (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)

        return FlaxCLIPOutput(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class FlaxCLIPModelTransfer(FlaxCLIPPreTrainedModel):
    module_class = FlaxCLIPModuleTransfer
