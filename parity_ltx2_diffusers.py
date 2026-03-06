import torch
import numpy as np
import diffusers.utils.torch_utils
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
        t = tensor.float()
        print(f"[{name}] min: {t.min().item():.5f}, max: {t.max().item():.5f}, mean: {t.mean().item():.5f}, std: {t.std().item():.5f}")

def get_hook(name):
    def hook(module, input, output):
        if hasattr(output, "latent_dist"):
            print_stat(name, output.latent_dist.sample())
        elif hasattr(output, "sample"):
            print_stat(name, output.sample)
        elif hasattr(output, "hidden_states") and output.hidden_states is not None:
             print_stat(name, output.hidden_states[-1])
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
    print_stat("connectors_video", output[0])
    print_stat("connectors_audio", output[1])

def hook_transformer(module, input, output):
    out = output if isinstance(output, tuple) else (output[0], output[1])
    print_stat("transformer_video", out[0])
    print_stat("transformer_audio", out[1])

def set_hooks(pipe):
    pipe.text_encoder.register_forward_hook(get_hook('text_encoder'))
    if hasattr(pipe, 'connectors'):
        pipe.connectors.register_forward_hook(hook_connectors)
    pipe.transformer.register_forward_hook(hook_transformer)
    pipe.vae.decoder.register_forward_hook(get_hook('vae_decoder'))
    if hasattr(pipe, 'audio_vae'):
        pipe.audio_vae.decoder.register_forward_hook(get_hook('audio_vae_decoder'))
    if hasattr(pipe, 'vocoder'):
        pipe.vocoder.register_forward_hook(get_hook('vocoder'))

def main():
    pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    
    set_hooks(pipe)

    prompt = "A man in a brightly lit room talks on a vintage telephone. In a low, heavy voice, he says, 'I understand. I won't call again. Goodbye.' He hangs up the receiver and looks down with a sad expression. He holds the black rotary phone to his right ear with his right hand, his left hand holding a rocks glass with amber liquid. He wears a brown suit jacket over a white shirt, and a gold ring on his left ring finger. His short hair is neatly combed, and he has light skin with visible wrinkles around his eyes. The camera remains stationary, focused on his face and upper body. The room is brightly lit by a warm light source off-screen to the left, casting shadows on the wall behind him. The scene appears to be from a dramatic movie."
    negative_prompt = "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."
    
    generator = torch.Generator(device=pipe.device).manual_seed(10)

    height = 512
    width = 768
    num_frames = 121
    frame_rate = 24.0

    print("Running pipeline...")
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
    
    video_noise = [t for t in randn_tensors_saved if len(t.shape) == 5][0]
    audio_noise = [t for t in randn_tensors_saved if len(t.shape) == 4][0]
    
    np.save("../maxdiffusion/video_noise.npy", video_noise)
    np.save("../maxdiffusion/audio_noise.npy", audio_noise)
    print("Saved video_noise.npy and audio_noise.npy to maxdiffusion folder.")

if __name__ == '__main__':
    main()
