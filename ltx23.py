import os
import sys
from typing import Any, Optional
import time
import numpy as np
import torch
import av

# Add src to path to import from local diffusers repo
sys.path.insert(0, os.path.abspath("src"))
from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline

# Helper functions for saving video with audio (copied from MaxDiffusion)
def _prepare_audio_stream(container, audio_sample_rate: int):
    from fractions import Fraction
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

    # flush audio encoder
    for packet in audio_stream.encode():
        container.mux(packet)

def _write_audio(
    container,
    audio_stream,
    samples: Any,
    audio_sample_rate: int,
    target_format: str = "s16",
) -> None:
    samples = np.asarray(samples)

    if samples.ndim == 1:
        samples = samples[:, None]

    if samples.shape[0] == 2 and samples.shape[1] != 2:
        samples = samples.T  # Now (Time, 2)

    if samples.shape[1] != 2:
        raise ValueError(f"Expected samples with 2 channels; got shape {samples.shape}.")

    if target_format == "s16":
        if samples.dtype != np.int16:
            samples = np.clip(samples, -1.0, 1.0)
            samples = (samples * 32767.0).astype(np.int16)
    elif target_format == "s32":
        if samples.dtype != np.int32:
            samples = np.clip(samples, -1.0, 1.0)
            samples = (samples * 2147483647.0).astype(np.int32)
    elif target_format in ["flt", "dbl", "fltp", "dblp"]:
        target_dtype = np.float32 if "flt" in target_format else np.float64
        if samples.dtype != target_dtype:
            samples = samples.astype(target_dtype)
    else:
        raise ValueError(f"Unsupported target_format for converting numpy array: {target_format}")

    samples_np = np.ascontiguousarray(samples).reshape(1, -1)

    frame_in = av.AudioFrame.from_ndarray(
        samples_np,
        format=target_format,
        layout="stereo",
    )
    frame_in.sample_rate = audio_sample_rate

    _resample_audio(container, audio_stream, frame_in)

def export_to_video_with_audio(
    video: Any, fps: int, audio: Optional[Any], audio_sample_rate: Optional[int], output_path: str, audio_format: str = "s16"
) -> None:
    video_np = np.asarray(video)

    if video_np.ndim == 4:
        _, height, width, _ = video_np.shape
    elif video_np.ndim == 5:
        video_np = video_np[0]
        _, height, width, _ = video_np.shape
    else:
        raise ValueError(f"export_to_video_with_audio expects a 4D or 5D video tensor, got {video_np.ndim}D")

    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    if audio is not None:
        if audio_sample_rate is None:
            raise ValueError("audio_sample_rate is required when audio is provided")
        audio_stream = _prepare_audio_stream(container, audio_sample_rate)

    for frame_array in video_np:
        frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    if audio is not None:
        _write_audio(container, audio_stream, audio, audio_sample_rate, target_format=audio_format)

    container.close()

# Configurations (matching ltx2_3_video.yml)
prompt = "A man in a brightly lit room talks on a vintage telephone. In a low, heavy voice, he says, 'I understand. I won't call again. Goodbye.' He hangs up the receiver and looks down with a sad expression. He holds the black rotary phone to his right ear with his right hand, his left hand holding a rocks glass with amber liquid. He wears a brown suit jacket over a white shirt, and a gold ring on his left ring finger. His short hair is neatly combed, and he has light skin with visible wrinkles around his eyes. The camera remains stationary, focused on his face and upper body. The room is brightly lit by a warm light source off-screen to the left, casting shadows on the wall behind him. The scene appears to be from a dramatic movie."
negative_prompt = "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."
height = 512
width = 768
num_frames = 121
num_inference_steps = 30
guidance_scale = 3.0
audio_guidance_scale = 7.0
stg_scale = 1.0
audio_stg_scale = 1.0
modality_scale = 1.0
audio_modality_scale = 1.0
spatio_temporal_guidance_blocks = [28]
fps = 24
seed = 10

print("Loading LTX2Pipeline...")
pipe = LTX2Pipeline.from_pretrained("dg845/LTX-2.3-Diffusers", torch_dtype=torch.bfloat16)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
pipe.to(device)

print("Generating...")
generator = torch.Generator(device).manual_seed(seed)
out = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    audio_guidance_scale=audio_guidance_scale,
    stg_scale=stg_scale,
    audio_stg_scale=audio_stg_scale,
    modality_scale=modality_scale,
    audio_modality_scale=audio_modality_scale,
    spatio_temporal_guidance_blocks=spatio_temporal_guidance_blocks,
    generator=generator,
    frame_rate=fps,
)

videos = out.frames
audios = out.audio

output_path = f"./ltx2_output_{seed}_0.mp4"

print(f"Saving output to {output_path}...")
export_to_video_with_audio(
    video=videos[0],
    fps=fps,
    audio=audios[0] if audios is not None else None,
    audio_sample_rate=24000,  # Default for LTX2 audio
    output_path=output_path,
)
print("Done!")
