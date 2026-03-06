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

def get_hook(name):
    def hook(module, input, output):
        if hasattr(output, "latent_dist"):
            print_stat(name, output.latent_dist.sample())
        elif hasattr(output, "sample"):
            print_stat(name, output.sample)
        elif hasattr(output, "hidden_states") and output.hidden_states is not None:
             t = output.hidden_states[-1].cpu().float().numpy()
             print_stat(name, t)
             if name == "text_encoder":
                 np.save("diffusers_text_encoder.npy", t)
        elif hasattr(output, "last_hidden_state"):
            print_stat(name, output.last_hidden_state)
        elif type(output).__name__ == "LTX2PipelineOutput":
            print_stat(name, output.frames)
        else:
            out = output[0] if isinstance(output, (list, tuple)) else output
            if isinstance(out, torch.Tensor):
                print_stat(name, out)
    return hook

def hook_connectors(module, input, output):
    print_stat("connectors_input", input[0])
    print_stat("connectors_video", output[0])
    print_stat("connectors_audio", output[1])

from diffusers.pipelines.ltx2 import pipeline_ltx2

def hook_transformer_pre(module, args, kwargs):
    hidden_states = kwargs.get("hidden_states")
    if hidden_states is None and len(args) > 0:
        hidden_states = args[0]
        
    audio_hidden_states = kwargs.get("audio_hidden_states")
    if audio_hidden_states is None and len(args) > 1:
        audio_hidden_states = args[1]
        
    timestep = kwargs.get("timestep")
    if timestep is None and len(args) > 4:
        timestep = args[4]
        
    if hidden_states is not None:
        print_stat("transformer_input_video_latents", hidden_states)
    if audio_hidden_states is not None:
        print_stat("transformer_input_audio_latents", audio_hidden_states)
    if timestep is not None:
        print_stat("transformer_timestep", timestep)

def hook_transformer(module, input, output):
    out = output if isinstance(output, tuple) else (output[0], output[1])
    print_stat("transformer_video", out[0])
    print_stat("transformer_audio", out[1])

def set_hooks(pipe):
    # Patch Transformer forward pass
    orig_transformer_forward = type(pipe.transformer).forward
    def patched_transformer_forward(self, *args, **kwargs):
        print("\n=== TRANSFORMER INPUTS ===")
        if "hidden_states" in kwargs:
             print_stat("transformer_input_video_latents", kwargs["hidden_states"])
        if "audio_hidden_states" in kwargs:
             print_stat("transformer_input_audio_latents", kwargs["audio_hidden_states"])
        if "timestep" in kwargs:
             print_stat("transformer_timestep", kwargs["timestep"])
        out = orig_transformer_forward(self, *args, **kwargs)
        print("\n=== TRANSFORMER OUTPUTS ===")
        
        if hasattr(out, "sample"):
            print_stat("transformer_video", out.sample)
            if hasattr(out, "audio_sample"):
                print_stat("transformer_audio", out.audio_sample)
        elif isinstance(out, (tuple, list)):
            print_stat("transformer_video", out[0])
            if len(out) > 1:
                print_stat("transformer_audio", out[1])
        else:
             print_stat("transformer_video", out)
        return out
    type(pipe.transformer).forward = patched_transformer_forward

    if hasattr(pipe, 'vae'):
        pipe.vae.decoder.register_forward_hook(get_hook('vae_decoder'))
    
    if hasattr(pipe, 'audio_vae'):
        pipe.audio_vae.decoder.register_forward_hook(get_hook('audio_vae_decoder'))

    if hasattr(pipe, 'vocoder'):
        pipe.vocoder.register_forward_hook(get_hook('vocoder'))

def main():
    pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    
    print("DIFFUSERS SCHEDULER CONFIG:", pipe.scheduler.config)
    prompt = "A man in a brightly lit room talks on a vintage telephone. In a low, heavy voice, he says, 'I understand. I won't call again. Goodbye.' He hangs up the receiver and looks down with a sad expression. He holds the black rotary phone to his right ear with his right hand, his left hand holding a rocks glass with amber liquid. He wears a brown suit jacket over a white shirt, and a gold ring on his left ring finger. His short hair is neatly combed, and he has light skin with visible wrinkles around his eyes. The camera remains stationary, focused on his face and upper body. The room is brightly lit by a warm light source off-screen to the left, casting shadows on the wall behind him. The scene appears to be from a dramatic movie."
    negative_prompt = "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."
    
    generator = torch.Generator(device=pipe.device).manual_seed(10)

    height = 512
    width = 768
    num_frames = 121
    frame_rate = 24.0

    print("Running pipeline...")
    print("\n--- DEBUG TOKENIZER DIFFUSERS ---")
    prompt = "A man in a brightly lit room talks on a vintage telephone. In a low, heavy voice, he says, 'I understand. I won't call again. Goodbye.' He hangs up the receiver and looks down with a sad expression. He holds the black rotary phone to his right ear with his right hand, his left hand holding a rocks glass with amber liquid. He wears a brown suit jacket over a white shirt, and a gold ring on his left ring finger. His short hair is neatly combed, and he has light skin with visible wrinkles around his eyes. The camera remains stationary, focused on his face and upper body. The room is brightly lit by a warm light source off-screen to the left, casting shadows on the wall behind him. The scene appears to be from a dramatic movie."
    negative_prompt = "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."
    
    # Just grab tokenizer directly from pipe
    # Gemma expects left padding for chat-style prompts
    pipe.tokenizer.padding_side = "left"
    if pipe.tokenizer.pad_token is None:
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
        
    for p_name, p_text in [("prompt", prompt), ("negative_prompt", negative_prompt)]:
        text_inputs = pipe.tokenizer(
            [p_text],
            padding="max_length",
            max_length=1024,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        print(f"{p_name} input_ids sum: {text_inputs.input_ids.sum().item()}, non-padded (attn_mask sum): {text_inputs.attention_mask.sum().item()}")
    print("---------------------------------\n")

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
    
    video_noise_tensors = [t for t in randn_tensors_saved if len(t.shape) == 5]
    audio_noise_tensors = [t for t in randn_tensors_saved if len(t.shape) == 4]
    
    if not video_noise_tensors:
         print("Warning: no 5D video noise tensor was saved!")
    else:
         video_noise = video_noise_tensors[0]
         np.save("../maxdiffusion/video_noise.npy", video_noise)
         print(f"Saved video_noise.npy with shape {video_noise.shape} to maxdiffusion folder.")

    if not audio_noise_tensors:
         print("Warning: no 4D audio noise tensor was saved!")
    else:
         audio_noise = audio_noise_tensors[0]
         np.save("../maxdiffusion/audio_noise.npy", audio_noise)
         print(f"Saved audio_noise.npy with shape {audio_noise.shape} to maxdiffusion folder.")

if __name__ == '__main__':
    main()
