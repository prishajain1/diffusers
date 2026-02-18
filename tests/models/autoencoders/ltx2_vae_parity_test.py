import sys
import os

# Prioritize the local git repository (../../../src) over the pip-installed environment
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

import torch
import numpy as np
from diffusers.models.autoencoders.autoencoder_kl_ltx2 import AutoencoderKLLTX2Video

def main():
    print("Initializing Diffusers LTX-2 VAE...")
    
    # Initialize with default/small configuration
    model = AutoencoderKLLTX2Video(
        in_channels=3,
        out_channels=3,
        latent_channels=128,
        block_out_channels=(256, 512, 1024, 2048),
        decoder_block_out_channels=(256, 512, 1024),
        layers_per_block=(4, 6, 6, 2, 2),
        decoder_layers_per_block=(5, 5, 5, 5),
    )
    
    model.eval()
    
    # Generate deterministic random input
    # Shape: (B, C, T, H, W) for PyTorch
    B, C, T, H, W = 1, 3, 9, 64, 64
    torch.manual_seed(42)
    sample = torch.rand((B, C, T, H, W)) * 2.0 - 1.0  # [-1, 1] range
    
    print(f"\n--- Input ---")
    print(f"Shape: {sample.shape}")
    print(f"Mean: {sample.mean().item():.6f}, Std: {sample.std().item():.6f}")
    
    # Run Encoder
    print("\nRunning Encoder...")
    with torch.no_grad():
        posterior = model.encode(sample).latent_dist
        # Sample or use mode. Using mode for determinism without extra RNG steps.
        # mode() uses just the mean.
        latents = posterior.mode()
        
    print(f"\n--- Encoder Latents ---")
    print(f"Shape: {latents.shape}")
    print(f"Mean: {latents.mean().item():.6f}, Std: {latents.std().item():.6f}")
    
    # Run Decoder
    print("\nRunning Decoder...")
    with torch.no_grad():
        reconstruction = model.decode(latents).sample
        
    print(f"\n--- Decoder Output ---")
    print(f"Shape: {reconstruction.shape}")
    print(f"Mean: {reconstruction.mean().item():.6f}, Std: {reconstruction.std().item():.6f}")

    # Tiled Passes
    print("\nRunning Tiled Encoder/Decoder Passes...")
    model.tile_sample_min_height = 32
    model.tile_sample_min_width = 32
    model.tile_sample_stride_height = 16
    model.tile_sample_stride_width = 16
    model.tile_latent_min_height = 2 
    model.tile_latent_min_width = 2  
    model.tile_latent_stride_height = 1
    model.tile_latent_stride_width = 1
    model.enable_tiling()
    
    with torch.no_grad():
        latents_tiled = model.encode(sample).latent_dist.mode()
        reconstruction_tiled = model.decode(latents_tiled).sample
        
    model.disable_tiling()
    
    # Temporal Tiled Passes
    print("\nRunning Temporal Tiled Encoder/Decoder Passes...")
    model.tile_sample_min_num_frames = 17 # Must be divisible by upscale mathematically
    model.tile_sample_stride_num_frames = 8
    model.use_framewise_decoding = True
    
    # Extend sample to verify temporal tiling logic
    B, C, T, H, W = 1, 3, 33, 64, 64
    torch.manual_seed(42)
    sample_temporal = torch.rand((B, C, T, H, W)) * 2.0 - 1.0
    
    with torch.no_grad():
        latents_temporal_tiled = model.encode(sample_temporal).latent_dist.mode()
        reconstruction_temporal_tiled = model.decode(latents_temporal_tiled).sample

    # Save to disk for MaxDiffusion
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "ltx2_parity_data"))
    os.makedirs(save_dir, exist_ok=True)
    
    # Save Model Weights
    print(f"\nSaving model weights to {save_dir}/pytorch_model.bin...")
    torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
    
    # Save Input and Outputs for comparison
    np.save(os.path.join(save_dir, "input.npy"), sample.numpy())
    np.save(os.path.join(save_dir, "latents.npy"), latents.numpy())
    np.save(os.path.join(save_dir, "reconstruction.npy"), reconstruction.numpy())
    
    np.save(os.path.join(save_dir, "latents_tiled.npy"), latents_tiled.numpy())
    np.save(os.path.join(save_dir, "reconstruction_tiled.npy"), reconstruction_tiled.numpy())
    
    np.save(os.path.join(save_dir, "input_temporal.npy"), sample_temporal.numpy())
    np.save(os.path.join(save_dir, "latents_temporal_tiled.npy"), latents_temporal_tiled.numpy())
    np.save(os.path.join(save_dir, "reconstruction_temporal_tiled.npy"), reconstruction_temporal_tiled.numpy())
    print("Done!")

if __name__ == "__main__":
    main()
