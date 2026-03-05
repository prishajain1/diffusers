import torch
from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline
from diffusers.utils import export_to_video
import os

def generate_video():
    # Load the pipeline
    # The reference LTX-2 model from Lightricks
    model_id = "Lightricks/LTX-2"
    
    print(f"Loading LTX2Pipeline from {model_id}...")
    pipeline = LTX2Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    
    # Move to GPU if available, else CPU (or MPS for Mac)
    if torch.cuda.is_available():
        pipeline = pipeline.to("cuda")
    elif torch.backends.mps.is_available():
        pipeline = pipeline.to("mps")
    else:
        pipeline = pipeline.to("cpu")

    # Generation parameters matching maxdiffusion's ltx2_video.yml
    prompt = "A man in a brightly lit room talks on a vintage telephone. In a low, heavy voice, he says, 'I understand. I won't call again. Goodbye.' He hangs up the receiver and looks down with a sad expression. He holds the black rotary phone to his right ear with his right hand, his left hand holding a rocks glass with amber liquid. He wears a brown suit jacket over a white shirt, and a gold ring on his left ring finger. His short hair is neatly combed, and he has light skin with visible wrinkles around his eyes. The camera remains stationary, focused on his face and upper body. The room is brightly lit by a warm light source off-screen to the left, casting shadows on the wall behind him. The scene appears to be from a dramatic movie."
    negative_prompt = "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."
    
    # In diffusers, num_frames is the number of latent frames if not fully decoded, 
    # but for LTXVideoPipeline it expects actual video frames.
    num_frames = 121
    height = 512
    width = 768
    num_inference_steps = 40
    guidance_scale = 3.0
    decode_timestep = 0.05
    decode_noise_scale = 0.025
    seed = 10

    print("Generating video...")
    
    # Set seed
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    
    # Generate
    output = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        decode_timestep=decode_timestep,
        decode_noise_scale=decode_noise_scale,
    )

    # Get the video frames
    video_frames = output.frames[0]
    
    # Save the video
    output_filename = "ltx2_diffusers_output.mp4"
    export_to_video(video_frames, output_filename, fps=24)
    print(f"Video saved to {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    generate_video()
