import torch
import unittest
import sys
import os
# Make sure we can import from src if running from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel, LTX2AudioVideoRotaryPosEmbed

class LTX2ParityTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 1
        self.num_frames = 4
        self.height = 32
        self.width = 32
        self.patch_size = 1
        self.patch_size_t = 1
        
        self.dim = 32
        self.in_channels = 8
        self.out_channels = 8
        self.audio_in_channels = 4
        
        # LTX-2 specific params
        self.caption_channels = 32
        self.cross_attention_dim = 1024
        self.audio_cross_attention_dim = 1024
        
        self.device = "cpu" 
        
        self.model = LTX2VideoTransformer3DModel(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            patch_size=self.patch_size,
            patch_size_t=self.patch_size_t,
            num_attention_heads=8,
            attention_head_dim=128, 
            num_layers=1,
            caption_channels=self.caption_channels,
            cross_attention_dim=self.cross_attention_dim,
            audio_in_channels=self.audio_in_channels,
            audio_out_channels=self.audio_in_channels,
            audio_num_attention_heads=8,
            audio_attention_head_dim=128,
            audio_cross_attention_dim=self.audio_cross_attention_dim,
            attention_bias=True,
            attention_out_bias=True,
        )
        self.model.to(self.device).eval()
        
    def test_rope_split_shape(self):
        """
        Verifies that LTX2AudioVideoRotaryPosEmbed with rope_type='split' returns (B, H, T, D//2).
        """
        print("\n=== Testing Diffusers LTX2AudioVideoRotaryPosEmbed Split Shape ===")
        dim = 1024
        heads = 32
        
        rope = LTX2AudioVideoRotaryPosEmbed(
            dim=dim,
            patch_size=1,
            patch_size_t=1,
            base_num_frames=8,
            base_height=32,
            base_width=32,
            modality="video",
            rope_type="split",
            num_attention_heads=heads
        ).to(self.device)
        
        batch_size = 1
        num_patches = 10
        grid = torch.randn(batch_size, 3, num_patches).to(self.device)
        
        cos, sin = rope(grid)
        
        print(f"Cos shape: {cos.shape}")
        
        self.assertEqual(cos.shape, (batch_size, heads, num_patches, 16))
        self.assertEqual(sin.shape, (batch_size, heads, num_patches, 16))


    def test_transformer_3d_model_forward(self):
        print("\n=== Testing Diffusers LTX2VideoTransformer3DModel Forward ===")
        
        # Flattened inputs required for LTX2VideoTransformer3DModel (proj_in is Linear)
        # Sequence length = L = (F/p_t * H/p_h * W/p_w)
        seq_len = (self.num_frames // self.patch_size_t) * (self.height // self.patch_size) * (self.width // self.patch_size)
        
        hidden_states = torch.zeros(
            (self.batch_size, seq_len, self.in_channels), 
            device=self.device
        )
        
        # Audio sequence length is fixed in test setup to 10 frames usually
        audio_seq_len = 128 
        audio_hidden_states = torch.zeros(
            (self.batch_size, audio_seq_len, self.audio_in_channels), 
            device=self.device
        )
        
        timestep = torch.tensor([1.0], device=self.device)
        # encoder_hidden_states has caption_channels dim
        encoder_hidden_states = torch.zeros((self.batch_size, 128, self.caption_channels), device=self.device)
        audio_encoder_hidden_states = torch.zeros((self.batch_size, 128, self.caption_channels), device=self.device)
        
        # Need attention masks? Usually optional.
        encoder_attention_mask = torch.ones((self.batch_size, 128), device=self.device)
        audio_encoder_attention_mask = torch.ones((self.batch_size, 128), device=self.device)
        
        with torch.no_grad():
            output = self.model(
                hidden_states=hidden_states,
                audio_hidden_states=audio_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                audio_encoder_hidden_states=audio_encoder_hidden_states,
                timestep=timestep,
                num_frames=self.num_frames,
                height=self.height,
                width=self.width,
                audio_num_frames=128, 
                fps=24.0,
                return_dict=True,
                encoder_attention_mask=encoder_attention_mask,
                audio_encoder_attention_mask=audio_encoder_attention_mask
            )

        sample = output.sample
        audio_sample = output.audio_sample
        
        print(f"Diffusers Output Video Shape: {sample.shape}")
        print(f"Diffusers Output Audio Shape: {audio_sample.shape}")
        
        # Expect (B, L, C) output
        self.assertEqual(sample.shape, hidden_states.shape)
        self.assertEqual(audio_sample.shape, audio_hidden_states.shape)

    def test_export_parity_data(self):
        """
        Exports the model state_dict and inputs/outputs for parity testing with MaxDiffusion.
        """
        print("\n=== Exporting Parity Data ===")
        # 1. Initialize logic matches setUp but ensuring deterministic(ish) state if needed
        # We use the existing self.model which is already initialized with random weights
        
        # 2. Prepare Inputs (same as other tests)
        # Flattened inputs required for LTX2VideoTransformer3DModel
        seq_len = (self.num_frames // self.patch_size_t) * (self.height // self.patch_size) * (self.width // self.patch_size)
        hidden_states = torch.randn(self.batch_size, seq_len, self.in_channels).to(self.device).to(torch.float32)
        
        audio_seq_len = 128
        audio_hidden_states = torch.randn(self.batch_size, audio_seq_len, self.audio_in_channels).to(self.device).to(torch.float32)
        
        encoder_hidden_states = torch.randn(self.batch_size, 128, self.caption_channels).to(self.device).to(torch.float32)
        audio_encoder_hidden_states = torch.randn(self.batch_size, 128, self.caption_channels).to(self.device).to(torch.float32)
        
        encoder_attention_mask = torch.ones(self.batch_size, 128).to(self.device).to(torch.int64)
        audio_encoder_attention_mask = torch.ones(self.batch_size, 128).to(self.device).to(torch.int64)
        
        timestep = torch.tensor([1.0]).to(self.device).to(torch.float32)
        
        # 3. Forward Pass
        with torch.no_grad():
            output = self.model(
                hidden_states=hidden_states,
                audio_hidden_states=audio_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                audio_encoder_hidden_states=audio_encoder_hidden_states,
                timestep=timestep,
                encoder_attention_mask=encoder_attention_mask,
                audio_encoder_attention_mask=audio_encoder_attention_mask,
                num_frames=self.num_frames,
                height=self.height,
                width=self.width,
                audio_num_frames=128,
                return_dict=True
            )
        
        print("\n=== Input Verification ===")
        print(f"Hidden States Sum: {hidden_states.sum().item()}")
        print(f"Audio Hidden States Sum: {audio_hidden_states.sum().item()}")
        print(f"Encoder Hidden States Sum: {encoder_hidden_states.sum().item()}")
        print(f"Audio Encoder Hidden States Sum: {audio_encoder_hidden_states.sum().item()}")
        print(f"Timestep: {timestep.item()}")
        print("==========================\n")
        
        print(f"Diffusers Sample Max: {output.sample.max()}")
        print(f"Diffusers Sample Min: {output.sample.min()}")
        print(f"Diffusers Sample Mean: {output.sample.mean()}")
        print(f"Diffusers Sample Std: {output.sample.std()}")
        
        print(f"Diffusers Audio Max: {output.audio_sample.max()}")
        print(f"Diffusers Audio Min: {output.audio_sample.min()}")
        print(f"Diffusers Audio Mean: {output.audio_sample.mean()}")
        print(f"Diffusers Audio Std: {output.audio_sample.std()}")
        
        # 4. Save Data
        parity_data = {
            "state_dict": self.model.state_dict(),
            "inputs": {
                "hidden_states": hidden_states,
                "audio_hidden_states": audio_hidden_states,
                "encoder_hidden_states": encoder_hidden_states,
                "audio_encoder_hidden_states": audio_encoder_hidden_states,
                "timestep": timestep,
                "encoder_attention_mask": encoder_attention_mask,
                "audio_encoder_attention_mask": audio_encoder_attention_mask
            },
            "outputs": {
                "sample": output.sample,
                "audio_sample": output.audio_sample
            },
            "config": {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "patch_size": self.patch_size,
                "head_dim": 128,
                "inner_dim": 1024,
                "caption_channels": self.caption_channels
            }
        }
        
        save_path = "ltx2_parity_data.pt"
        torch.save(parity_data, save_path)
        print(f"Saved parity data to {save_path}")
        print(f"Sample output mean: {output.sample.mean().item()}")
        print(f"Sample output std: {output.sample.std().item()}")

    def test_transformer_3d_model_forward_split(self):
        print("\n=== Testing Diffusers LTX2VideoTransformer3DModel Forward (Split RoPE) ===")
        
        # Instantiate model with rope_type="split"
        model_split = LTX2VideoTransformer3DModel(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            patch_size=self.patch_size,
            patch_size_t=self.patch_size_t,
            num_attention_heads=8,
            attention_head_dim=128, 
            num_layers=1,
            caption_channels=self.caption_channels,
            cross_attention_dim=self.cross_attention_dim,
            audio_in_channels=self.audio_in_channels,
            audio_out_channels=self.audio_in_channels,
            audio_num_attention_heads=8,
            audio_attention_head_dim=128,
            audio_cross_attention_dim=self.audio_cross_attention_dim,
            attention_bias=True,
            attention_out_bias=True,
            rope_type="split"
        )
        model_split.to(self.device).eval()
        
        # Inputs
        seq_len = (self.num_frames // self.patch_size_t) * (self.height // self.patch_size) * (self.width // self.patch_size)
        hidden_states = torch.randn(self.batch_size, seq_len, self.in_channels).to(self.device)
        audio_seq_len = 128 
        audio_hidden_states = torch.randn(self.batch_size, audio_seq_len, self.audio_in_channels).to(self.device)
        timestep = torch.tensor([1.0], device=self.device)
        encoder_hidden_states = torch.randn((self.batch_size, 128, self.caption_channels), device=self.device)
        audio_encoder_hidden_states = torch.randn((self.batch_size, 128, self.caption_channels), device=self.device)
        encoder_attention_mask = torch.ones((self.batch_size, 128), device=self.device)
        audio_encoder_attention_mask = torch.ones((self.batch_size, 128), device=self.device)
        
        with torch.no_grad():
            output = model_split(
                hidden_states=hidden_states,
                audio_hidden_states=audio_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                audio_encoder_hidden_states=audio_encoder_hidden_states,
                timestep=timestep,
                num_frames=self.num_frames,
                height=self.height,
                width=self.width,
                audio_num_frames=128, 
                fps=24.0,
                return_dict=True,
                encoder_attention_mask=encoder_attention_mask,
                audio_encoder_attention_mask=audio_encoder_attention_mask
            )

        print(f"Diffusers Output Video Shape (Split): {output.sample.shape}")
        self.assertEqual(output.sample.shape, hidden_states.shape)
        self.assertEqual(output.audio_sample.shape, audio_hidden_states.shape)

    def test_export_parity_data_split(self):
        """
        Exports the model state_dict and inputs/outputs for parity testing with MaxDiffusion (Split RoPE).
        """
        print("\n=== Exporting Parity Data (Split RoPE) ===")
        
        model_split = LTX2VideoTransformer3DModel(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            patch_size=self.patch_size,
            patch_size_t=self.patch_size_t,
            num_attention_heads=8,
            attention_head_dim=128, 
            num_layers=1,
            caption_channels=self.caption_channels,
            cross_attention_dim=self.cross_attention_dim,
            audio_in_channels=self.audio_in_channels,
            audio_out_channels=self.audio_in_channels,
            audio_num_attention_heads=8,
            audio_attention_head_dim=128,
            audio_cross_attention_dim=self.audio_cross_attention_dim,
            attention_bias=True,
            attention_out_bias=True,
            rope_type="split"
        )
        model_split.to(self.device).eval()

        # Prepare Inputs
        seq_len = (self.num_frames // self.patch_size_t) * (self.height // self.patch_size) * (self.width // self.patch_size)
        hidden_states = torch.randn(self.batch_size, seq_len, self.in_channels).to(self.device).to(torch.float32)
        audio_seq_len = 128
        audio_hidden_states = torch.randn(self.batch_size, audio_seq_len, self.audio_in_channels).to(self.device).to(torch.float32)
        encoder_hidden_states = torch.randn(self.batch_size, 128, self.caption_channels).to(self.device).to(torch.float32)
        audio_encoder_hidden_states = torch.randn(self.batch_size, 128, self.caption_channels).to(self.device).to(torch.float32)
        encoder_attention_mask = torch.ones(self.batch_size, 128).to(self.device).to(torch.int64)
        audio_encoder_attention_mask = torch.ones(self.batch_size, 128).to(self.device).to(torch.int64)
        timestep = torch.tensor([1.0]).to(self.device).to(torch.float32)
        
        with torch.no_grad():
            output = model_split(
                hidden_states=hidden_states,
                audio_hidden_states=audio_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                audio_encoder_hidden_states=audio_encoder_hidden_states,
                timestep=timestep,
                encoder_attention_mask=encoder_attention_mask,
                audio_encoder_attention_mask=audio_encoder_attention_mask,
                num_frames=self.num_frames,
                height=self.height,
                width=self.width,
                audio_num_frames=128,
                return_dict=True
            )
        
        print("\n=== Input Verification (Split) ===")
        print(f"Hidden States Sum: {hidden_states.sum().item()}")
        
        # Save Data
        parity_data = {
            "state_dict": model_split.state_dict(),
            "inputs": {
                "hidden_states": hidden_states,
                "audio_hidden_states": audio_hidden_states,
                "encoder_hidden_states": encoder_hidden_states,
                "audio_encoder_hidden_states": audio_encoder_hidden_states,
                "timestep": timestep,
                "encoder_attention_mask": encoder_attention_mask,
                "audio_encoder_attention_mask": audio_encoder_attention_mask
            },
            "outputs": {
                "sample": output.sample,
                "audio_sample": output.audio_sample
            },
            "config": {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "patch_size": self.patch_size,
                "head_dim": 128,
                "inner_dim": 1024,
                "caption_channels": self.caption_channels,
                "rope_type": "split"
            }
        }
        
        save_path = "ltx2_parity_data_split.pt"
        torch.save(parity_data, save_path)
        print(f"Saved split parity data to {save_path}")

if __name__ == "__main__":
    unittest.main()

