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
print(f"Using device: {device}", flush=True)

# Load pipeline
from huggingface_hub import hf_hub_download
import json

config_file = hf_hub_download(repo_id="dg845/LTX-2.3-Diffusers", filename="config.json", subfolder="transformer")

with open(config_file, "r") as f:
    config_dict = json.load(f)

# Ensure configuration matches the parameters we discovered
config_dict["video_cross_attn_adaln"] = True
config_dict["audio_cross_attn_adaln"] = True

with open(config_file, "w") as f:
    json.dump(config_dict, f, indent=2)

print(f"Modified cached config file at: {config_file}", flush=True)

pipe = LTX2Pipeline.from_pretrained("dg845/LTX-2.3-Diffusers", torch_dtype=torch.bfloat16)
pipe.to(device)

prompt = (
    "A man in a brightly lit room talks on a vintage telephone. In a low, heavy voice, he says, "
    "'I understand. I won't call again. Goodbye.' He hangs up the receiver and looks down with a "
    "sad expression. He holds the black rotary phone to his right ear with his right hand, his left "
    "hand holding a rocks glass with amber liquid. He wears a brown suit jacket over a white shirt, "
    "and a gold ring on his left ring finger. His short hair is neatly combed, and he has light skin "
    "with visible wrinkles around his eyes. The camera remains stationary, focused on his face and "
    "upper body. The room is brightly lit by a warm light source off-screen to the left, casting "
    "shadows on the wall behind him. The scene appears to be from a dramatic movie."
)
negative_prompt = "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."

# Noise Save/Load Policy
home_dir = os.path.expanduser("~")
pt_latents_path = os.path.join(home_dir, "pt_latents_step_0.pt")
pt_audio_latents_path = os.path.join(home_dir, "pt_audio_latents_step_0.pt")

# We generate/determine shapes:
# Video unpacked: [B, C, T, H, W] = [1, 128, 16, 64, 96]
# Audio unpacked: [B, C, L, M] = [1, 8, 161, 16] (Mel bins 128 // 8 = 16, Latent channels = 8, Length = 161)
video_shape = (1, 128, 16, 64, 96)
audio_shape = (1, 8, 161, 16)

generator = torch.Generator(device=device).manual_seed(10)

if os.path.exists(pt_latents_path) and os.path.exists(pt_audio_latents_path):
    print("👉 Loading existing starting noise from home directory...", flush=True)
    latents = torch.load(pt_latents_path, weights_only=False).to(device=device, dtype=torch.bfloat16)
    audio_latents = torch.load(pt_audio_latents_path, weights_only=False).to(device=device, dtype=torch.bfloat16)
else:
    print("👉 Generating new starting noise and saving to home directory...", flush=True)
    latents = torch.randn(video_shape, generator=generator, device=device, dtype=torch.bfloat16)
    audio_latents = torch.randn(audio_shape, generator=generator, device=device, dtype=torch.bfloat16)
    torch.save(latents.cpu(), pt_latents_path)
    torch.save(audio_latents.cpu(), pt_audio_latents_path)
    print(f"Saved {pt_latents_path} and {pt_audio_latents_path}", flush=True)

print(f"Latents shape: {latents.shape} | mean: {latents.mean().item():.6f} | min: {latents.min().item():.6f} | max: {latents.max().item():.6f} | std: {latents.std().item():.6f}", flush=True)
print(f"Audio Latents shape: {audio_latents.shape} | mean: {audio_latents.mean().item():.6f} | min: {audio_latents.min().item():.6f} | max: {audio_latents.max().item():.6f} | std: {audio_latents.std().item():.6f}", flush=True)

# Run inference
print("🚀 Starting PyTorch pipeline inference (30 steps)...", flush=True)
with torch.no_grad():
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=1,
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
        decode_timestep=0.05,
        decode_noise_scale=0.025,
    )

print("✅ Inference completed!", flush=True)

from diffusers.pipelines.ltx2.export_utils import encode_video

video_path = os.path.join(script_dir, "output_ltx23.mp4")
print(f"Saving generated video and audio to {video_path}...", flush=True)

encode_video(
    video=output.frames[0] if isinstance(output.frames[0], list) else output.frames,
    fps=24,
    audio=output.audio[0],
    audio_sample_rate=48000,
    output_path=video_path
)

print(f"🎉 Video and audio saved successfully at {video_path}!", flush=True)
