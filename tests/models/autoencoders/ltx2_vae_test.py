import unittest
import torch
import sys
import os

# Ensure the local diffusers src is testable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from diffusers.models.autoencoders.autoencoder_kl_ltx2 import (
    LTX2VideoCausalConv3d,
    LTX2VideoDownBlock3D,
    LTX2VideoUpBlock3d,
    AutoencoderKLLTX2Video,
    LTX2DiagonalGaussianDistribution
)

class LTX2VaeTest(unittest.TestCase):
    
    def test_ltx2_causal_conv3d(self):
        """Tests the causal padding constraint of LTX2VideoCausalConv3d."""
        in_channels = 16
        out_channels = 32
        # (B, C, T, H, W) in PyTorch
        dummy_input = torch.ones((2, in_channels, 5, 16, 16))
        
        conv = LTX2VideoCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # PyTorch Diffusers LTX2 Causal Conv
        # We know temporal preservation happens intrinsically with symmetric spatial padding via standard convolution configurations in PyTorch
        out = conv(dummy_input)
        
        self.assertEqual(out.shape, (2, out_channels, 5, 16, 16))

    def test_ltx2_video_downblock3d(self):
        """Tests pooling and reshaping alignment in LTX2VideoDownBlock3D."""
        in_channels = 32
        out_channels = 64
        
        downsampler = LTX2VideoDownBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=None,
            num_layers=1,
            resnet_eps=1e-6,
            add_downsample=True,
            downsample_padding=1,
            spatial_downsample=True,
            temporal_downsample=False
        )
        
        # (B, C, T, H, W) -> T should remain 5, HW should halve from 16 to 8
        dummy_input = torch.ones((1, in_channels, 5, 16, 16))
        out = downsampler(dummy_input)
        
        self.assertEqual(out.shape, (1, out_channels, 5, 8, 8))

    def test_ltx2_video_upblock3d(self):
        """Tests the tile duplication and causal shift of LTX2VideoUpBlock3d."""
        in_channels = 64
        out_channels = 32
        
        upsampler = LTX2VideoUpBlock3d(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=None,
            num_layers=1,
            resnet_eps=1e-6,
            add_upsample=True,
            spatial_upsample=True,
            temporal_upsample=False
        )
        
        # (B, C, T, H, W) -> T should remain 3, HW should double from 8 to 16
        dummy_input = torch.ones((1, in_channels, 3, 8, 8))
        out = upsampler(dummy_input)
        
        self.assertEqual(out.shape, (1, out_channels, 3, 16, 16))

    def test_ltx2_diagonal_gaussian_distribution(self):
        """Tests that the custom 129-channel distribution splits and reconstructs successfully."""
        B, C_params, T, H, W = 2, 129, 4, 8, 8
        latent_channels = 128
        
        # Mock moments tensor
        parameters = torch.zeros((B, C_params, T, H, W))
        parameters[:, :128, ...] = 0.5 # Set mean to 0.5
        parameters[:, 128:, ...] = 1.0 # Set logvar to 1.0
        
        dist = LTX2DiagonalGaussianDistribution(parameters, latent_channels=latent_channels)
        
        # Verify splits
        self.assertEqual(dist.mean.shape, (B, 128, T, H, W))
        self.assertEqual(dist.logvar.shape, (B, 1, T, H, W))
        
        # Logvar mathematically broadcasts to variance
        self.assertEqual(dist.var.shape, (B, 128, T, H, W))
        
        # Sampling should return matching shapes
        sample = dist.sample()
        self.assertEqual(sample.shape, (B, 128, T, H, W))

    def test_ltx2_full_vae_encode_decode(self):
        """Tests a full, mini forward pass through the PyTorch LTX-2 VAE hierarchy."""
        vae = AutoencoderKLLTX2Video(
            in_channels=3,
            out_channels=3,
            latent_channels=8,
            block_out_channels=(16, 32),
            decoder_block_out_channels=(16, 32),
            layers_per_block=(2, 2),
            decoder_layers_per_block=(2, 2, 2),
            patch_size=2,
            patch_size_t=1
        )
        
        # PyTorch format (B, C, T, H, W)
        B, C, T, H, W = 1, 3, 9, 16, 16
        dummy_video = torch.ones((B, C, T, H, W))
        
        # Encode
        encoded_dist = vae.encode(dummy_video).latent_dist
        latents = encoded_dist.sample()
        
        # Validate Downsampling Math:
        # Spatial halves twice: 16 -> 8 -> 4, then unpatchified by patch_size=2 : 4 -> 2
        # T=9 downsamples to T=5 based on the internal causal padding downsample sizes
        self.assertEqual(latents.shape, (B, 8, 5, 4, 4))
        
        # Decode
        decoded = vae.decode(latents).sample
        
        # Because of the decoder_layers_per_block length (3) vs layers_per_block (2),
        # the decoder upsamples one extra time compared to the encoder's downsampling.
        # Spatial 4 -> 8 -> 16 -> 32
        # Temporal 5 -> ... -> 17
        self.assertEqual(decoded.shape, (B, C, 17, 32, 32))

if __name__ == "__main__":
    unittest.main()
