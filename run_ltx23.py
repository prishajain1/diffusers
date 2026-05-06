import sys
import os

# Dynamically resolve the local Diffusers src path relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "src"))

import torch
from diffusers import LTX2Pipeline
import numpy as np

# Ensure we use CPU if CUDA is not available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load pipeline
from huggingface_hub import hf_hub_download
import json

config_file = hf_hub_download(repo_id="dg845/LTX-2.3-Diffusers", filename="config.json", subfolder="transformer")

with open(config_file, "r") as f:
    config_dict = json.load(f)

config_dict["video_cross_attn_adaln"] = True
config_dict["audio_cross_attn_adaln"] = True

with open(config_file, "w") as f:
    json.dump(config_dict, f, indent=2)

print(f"Modified cached config file at: {config_file}")

pipe = LTX2Pipeline.from_pretrained("dg845/LTX-2.3-Diffusers", torch_dtype=torch.bfloat16)
pipe.to(device)

prompt = "A dog running on the left of a bicycle"
negative_prompt = "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."

# Load initial latents from MaxDiffusion
home_dir = os.path.expanduser("~")
latents = torch.load(os.path.join(home_dir, "latents_jax.pt")).to(device=device, dtype=torch.bfloat16)
audio_latents = torch.load(os.path.join(home_dir, "audio_latents_jax.pt")).to(device=device, dtype=torch.bfloat16)

print(f"Loaded latents shape: {latents.shape}")
print(f"Loaded audio_latents shape: {audio_latents.shape}")

print(f"WEIGHT DEBUG: block 0 to_q weight mean: {pipe.transformer.transformer_blocks[0].attn1.to_q.weight.mean().item():.6f}")

# Video-to-Audio Attention weights
v2a = pipe.transformer.transformer_blocks[0].video_to_audio_attn
print(f"WEIGHT DEBUG: block 0 v2a to_q mean: {v2a.to_q.weight.mean().item():.6f}, std: {v2a.to_q.weight.std().item():.6f}")
print(f"WEIGHT DEBUG: block 0 v2a to_k mean: {v2a.to_k.weight.mean().item():.6f}, std: {v2a.to_k.weight.std().item():.6f}")
print(f"WEIGHT DEBUG: block 0 v2a to_v mean: {v2a.to_v.weight.mean().item():.6f}, std: {v2a.to_v.weight.std().item():.6f}")
print(f"WEIGHT DEBUG: block 0 v2a to_out mean: {v2a.to_out[0].weight.mean().item():.6f}, std: {v2a.to_out[0].weight.std().item():.6f}")

# Audio-to-Video Attention weights
a2v = pipe.transformer.transformer_blocks[0].audio_to_video_attn
print(f"WEIGHT DEBUG: block 0 a2v to_q mean: {a2v.to_q.weight.mean().item():.6f}, std: {a2v.to_q.weight.std().item():.6f}")
print(f"WEIGHT DEBUG: block 0 a2v to_k mean: {a2v.to_k.weight.mean().item():.6f}, std: {a2v.to_k.weight.std().item():.6f}")
print(f"WEIGHT DEBUG: block 0 a2v to_v mean: {a2v.to_v.weight.mean().item():.6f}, std: {a2v.to_v.weight.std().item():.6f}")
print(f"WEIGHT DEBUG: block 0 a2v to_out mean: {a2v.to_out[0].weight.mean().item():.6f}, std: {a2v.to_out[0].weight.std().item():.6f}")

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
        guidance_rescale=0.7,
        audio_guidance_rescale=0.7,
        use_cross_timestep=True,
    )

print("Inference completed!")
