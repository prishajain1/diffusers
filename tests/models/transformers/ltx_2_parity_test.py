import torch
import unittest
import sys
import os
# Make sure we can import from src if running from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

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

if __name__ == "__main__":
    unittest.main()
