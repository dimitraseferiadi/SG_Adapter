import torch

# Load your pipeline (so you have CLIP text_encoder + tokenizer ready)
from pipeline_stable_diffusion_text_sg import StableDiffusionTextSGPipeline
pipeline = StableDiffusionTextSGPipeline.from_pretrained(
    args.pretrained_model_name_or_path,
    vae=accelerator.unwrap_model(vae),
    text_encoder=accelerator.unwrap_model(text_encoder),
    tokenizer=tokenizer,
    unet=accelerator.unwrap_model(unet),
    safety_checker=None,
    revision=args.revision,
    torch_dtype=weight_dtype,
)

# Import the new extractor
from utils import preprocess

# Example caption
caption = "a man riding a horse in the field"

# Call the new neural scene graph embedder
sg_embed = preprocess.extract_sg_embed_neural(
    objects=None,               # optional, will fallback to caption
    relations=None,
    text_encoder=pipe.text_encoder,
    tokenizer=pipe.tokenizer,
    caption=caption,
    max_relation_per_image=5,   # choose smaller for debugging
    device=pipe.text_encoder.device
)

print("Scene graph embedding shape:", sg_embed.shape)
print("dtype:", sg_embed.dtype)
print("sample values:", sg_embed[0, 0, :5])
