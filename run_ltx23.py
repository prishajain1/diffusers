import torch
from diffusers import LTX2Pipeline
import os

# Ensure we use CPU if CUDA is not available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load pipeline
pipe = LTX2Pipeline.from_pretrained("dg845/LTX-2.3-Diffusers", torch_dtype=torch.bfloat16)
pipe.to(device)

prompt = "A dog running on the left of a bicycle"
negative_prompt = "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."

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
    )

print("Inference completed!")
