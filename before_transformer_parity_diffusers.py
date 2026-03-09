import os
import sys

# Prioritize local src directory
sys.path.insert(0, os.path.abspath("src"))

import torch
import numpy as np

# Import diffusers directly, it will use the one from src/
import diffusers
import diffusers.utils.torch_utils
import diffusers.pipelines.ltx2.pipeline_ltx2
from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline
from diffusers.pipelines.ltx2.pipeline_ltx2 import randn_tensor as orig_randn_tensor

# Hook randn_tensor to save latents
randn_tensors_saved = []

def custom_randn_tensor(shape, generator=None, device=None, dtype=None, layout=None):
    t = orig_randn_tensor(shape, generator=generator, device=device, dtype=dtype, layout=layout)
    if len(shape) in (4, 5): 
        randn_tensors_saved.append(t.cpu().float().numpy())
    return t

diffusers.utils.torch_utils.randn_tensor = custom_randn_tensor
diffusers.pipelines.ltx2.pipeline_ltx2.randn_tensor = custom_randn_tensor

def print_stat(name, tensor):
    if hasattr(tensor, "shape"): 
        if hasattr(tensor, "cpu"):
             tensor = tensor.detach().cpu().float().numpy()
        t_np = np.array(tensor, dtype=np.float32)
        print(f"[{name}] shape: {t_np.shape}, min: {t_np.min():.5f}, max: {t_np.max():.5f}, mean: {t_np.mean():.5f}, std: {t_np.std():.5f}")

def hook_connectors(module, input, output):
    print("\n=== CONNECTORS OUTPUTS ===")
    if isinstance(output, (tuple, list)):
        print_stat("connectors_video", output[0])
        if len(output) > 1:
            print_stat("connectors_audio", output[1])
    else:
        print_stat("connectors_out", output)

def main():
    pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Patch Transformer forward pass to intercept inputs and EXIT EARLY
    orig_transformer_forward = type(pipe.transformer).forward
    def patched_transformer_forward(self, *args, **kwargs):
        print("\n=== TRANSFORMER INPUTS (DIFFUSERS) ===")
        if "hidden_states" in kwargs:
             print_stat("transformer_input_video_latents", kwargs["hidden_states"])
        if "audio_hidden_states" in kwargs:
             print_stat("transformer_input_audio_latents", kwargs["audio_hidden_states"])
        if "encoder_hidden_states" in kwargs:
             print_stat("transformers_encoder_hidden_states", kwargs["encoder_hidden_states"])
        if "audio_encoder_hidden_states" in kwargs:
             print_stat("transformers_audio_encoder_hidden_states", kwargs["audio_encoder_hidden_states"])
        if "timestep" in kwargs:
             print_stat("transformer_timestep", kwargs["timestep"])
        
        # Save latents to be loaded by maxdiffusion
        video_noise_tensors = [t for t in randn_tensors_saved if len(t.shape) == 5]
        audio_noise_tensors = [t for t in randn_tensors_saved if len(t.shape) == 4]
        
        if video_noise_tensors:
             video_noise = video_noise_tensors[0]
             np.save("../maxdiffusion/video_noise.npy", video_noise)
             print(f"Saved video_noise.npy with shape {video_noise.shape} to maxdiffusion folder.")
        
        if audio_noise_tensors:
             audio_noise = audio_noise_tensors[0]
             np.save("../maxdiffusion/audio_noise.npy", audio_noise)
             print(f"Saved audio_noise.npy with shape {audio_noise.shape} to maxdiffusion folder.")
             
        print("\n[SUCCESS] Captured all inputs up to Transformer logic. Exiting early to save compute.\n")
        import os
        os._exit(0)
    
    type(pipe.transformer).forward = patched_transformer_forward
    if hasattr(pipe, 'connectors'):
        pipe.connectors.register_forward_hook(hook_connectors)
    
    prompt = "A man in a brightly lit room talks on a vintage telephone. In a low, heavy voice, he says, 'I understand. I won't call again. Goodbye.' He hangs up the receiver and looks down with a sad expression. He holds the black rotary phone to his right ear with his right hand, his left hand holding a rocks glass with amber liquid. He wears a brown suit jacket over a white shirt, and a gold ring on his left ring finger. His short hair is neatly combed, and he has light skin with visible wrinkles around his eyes. The camera remains stationary, focused on his face and upper body. The room is brightly lit by a warm light source off-screen to the left, casting shadows on the wall behind him. The scene appears to be from a dramatic movie."
    negative_prompt = "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."
    
    generator = torch.Generator(device=pipe.device).manual_seed(10)

    height = 512
    width = 768
    num_frames = 121
    frame_rate = 24.0

    print("Running pipeline...")
    pipe.tokenizer.padding_side = "left"
    if pipe.tokenizer.pad_token is None:
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
        
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        num_inference_steps=40,
        guidance_scale=3.0,
        generator=generator,
        output_type="np",
        return_dict=False
    )

if __name__ == '__main__':
    main()
