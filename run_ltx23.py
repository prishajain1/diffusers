import torch
from diffusers import LTX2Pipeline
import os
import numpy as np

# Ensure we use CPU if CUDA is not available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load pipeline
from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

transformer = LTX2VideoTransformer3DModel.from_pretrained("dg845/LTX-2.3-Diffusers", subfolder="transformer", video_cross_attn_adaln=True, audio_cross_attn_adaln=True, torch_dtype=torch.bfloat16)
pipe = LTX2Pipeline.from_pretrained("dg845/LTX-2.3-Diffusers", transformer=transformer, torch_dtype=torch.bfloat16)
pipe.to(device)

prompt = "A dog running on the left of a bicycle"
negative_prompt = "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."

# Load initial latents from MaxDiffusion
home_dir = os.path.expanduser("~")
latents_jax = np.load(os.path.join(home_dir, "latents_jax.npy"))
audio_latents_jax = np.load(os.path.join(home_dir, "audio_latents_jax.npy"))

# Convert to torch tensors
latents = torch.from_numpy(latents_jax).to(device=device, dtype=torch.bfloat16)
audio_latents = torch.from_numpy(audio_latents_jax).to(device=device, dtype=torch.bfloat16)

# Transpose video latents from MaxDiffusion (B, F, H, W, C) to Diffusers (B, C, F, H, W)
latents = latents.permute(0, 4, 1, 2, 3)

print(f"Loaded latents shape: {latents.shape}")
print(f"Loaded audio_latents shape: {audio_latents.shape}")

# Run inference
print("Starting inference...")
with torch.no_grad():
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=40,
        guidance_scale=3.0,
        stg_scale=1.0,
        modality_scale=3.0,
        audio_guidance_scale=7.0,
        spatio_temporal_guidance_blocks=[28],
        num_frames=121,
        height=512,
        width=768,
        latents=latents,
        audio_latents=audio_latents,
    )

print("Inference completed!")
