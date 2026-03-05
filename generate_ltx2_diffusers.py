import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import torch
from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline
import os
import av
import numpy as np
from fractions import Fraction

def _prepare_audio_stream(container, audio_sample_rate: int):
    audio_stream = container.add_stream("aac", rate=audio_sample_rate)
    audio_stream.codec_context.sample_rate = audio_sample_rate
    audio_stream.codec_context.layout = "stereo"
    audio_stream.codec_context.time_base = Fraction(1, audio_sample_rate)
    return audio_stream

def _resample_audio(container, audio_stream, frame_in) -> None:
    cc = audio_stream.codec_context
    target_format = cc.format or "fltp"
    target_layout = cc.layout or "stereo"
    target_rate = cc.sample_rate or frame_in.sample_rate

    audio_resampler = av.audio.resampler.AudioResampler(
        format=target_format,
        layout=target_layout,
        rate=target_rate,
    )

    audio_next_pts = 0
    for rframe in audio_resampler.resample(frame_in):
        if rframe.pts is None:
            rframe.pts = audio_next_pts
        audio_next_pts += rframe.samples
        rframe.sample_rate = frame_in.sample_rate
        container.mux(audio_stream.encode(rframe))

    for packet in audio_stream.encode():
        container.mux(packet)

def _write_audio(container, audio_stream, samples: torch.Tensor, audio_sample_rate: int) -> None:
    if hasattr(samples, "cpu"):
        samples = samples.to(torch.float32).contiguous().cpu().numpy()
        
    if samples.ndim == 1:
        samples = samples[:, None]

    if samples.shape[0] == 2 and samples.shape[1] != 2:
        samples = samples.T

    if samples.shape[1] != 2:
        raise ValueError(f"Expected samples with 2 channels; got shape {samples.shape}.")

    if samples.dtype != np.int16:
        samples = np.clip(samples, -1.0, 1.0)
        samples = (samples * 32767.0).astype(np.int16)

    samples_np = np.ascontiguousarray(samples).reshape(1, -1)
        
    frame_in = av.AudioFrame.from_ndarray(
        samples_np,
        format="s16",
        layout="stereo",
    )
    frame_in.sample_rate = audio_sample_rate

    _resample_audio(container, audio_stream, frame_in)

def encode_video_with_audio(video, fps, audio, audio_sample_rate, output_path):
    if hasattr(video, "cpu"):
        video_np = video.cpu().numpy()
    elif isinstance(video, list):
        video_np = np.stack([np.array(v) for v in video])
    else:
        video_np = np.array(video)

    if video_np.ndim == 4:
        _, height, width, _ = video_np.shape
    elif video_np.ndim == 5:
        video_np = video_np[0]
        _, height, width, _ = video_np.shape

    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    if audio is not None:
        audio_stream = _prepare_audio_stream(container, audio_sample_rate)

    for frame_array in video_np:
        if isinstance(frame_array, np.ndarray) and frame_array.dtype != np.uint8:
            frame_array = (frame_array * 255).astype(np.uint8)
        frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    if audio is not None:
        _write_audio(container, audio_stream, audio, audio_sample_rate)

    container.close()
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
    audio_frames = output.audio[0] if output.audio is not None else None
    audio_sample_rate = getattr(pipeline.vocoder.config, "output_sampling_rate", 24000)
    encode_video_with_audio(video_frames, 24, audio_frames, audio_sample_rate, output_filename)
    print(f"Video saved to {os.path.abspath(output_filename)}")
if __name__ == "__main__":
    generate_video()
