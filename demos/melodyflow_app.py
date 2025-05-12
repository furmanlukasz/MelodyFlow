# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under thmage license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import logging
import os
import re
import shutil
import subprocess as sp
import sys
import time
import typing as tp
import uuid
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile

import gradio as gr
import spaces
import torch
from audiocraft.data.audio import audio_read, audio_write
from audiocraft.data.audio_utils import convert_audio
from audiocraft.models import MelodyFlow

MODEL = None  # Last used model
SPACE_ID = os.environ.get('SPACE_ID', '')
MODEL_PREFIX = os.environ.get('MODEL_PREFIX', 'facebook/')
IS_HF_SPACE = (MODEL_PREFIX + "MelodyFlow") in SPACE_ID
MAX_BATCH_SIZE = 12
N_REPEATS = 4
INTERRUPTING = False
MBD = None
# We have to wrap subprocess call to clean a bit the log when using gr.make_waveform
_old_call = sp.call

EULER = "euler"
MIDPOINT = "midpoint"

# Create storage directory for generated files
STORAGE_DIR = Path("generated_melodies")
STORAGE_DIR.mkdir(exist_ok=True)


def interrupt():
    global INTERRUPTING
    INTERRUPTING = True


class FileCleaner:
    def __init__(self, file_lifetime: float = 3600):
        self.file_lifetime = file_lifetime
        self.files = []

    def add(self, path: tp.Union[str, Path]):
        self._cleanup()
        self.files.append((time.time(), Path(path)))

    def _cleanup(self):
        now = time.time()
        for time_added, path in list(self.files):
            if now - time_added > self.file_lifetime:
                if path.exists():
                    path.unlink()
                self.files.pop(0)
            else:
                break


file_cleaner = FileCleaner()


def make_waveform(*args, **kwargs):
    # Further remove some warnings.
    be = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = gr.make_waveform(*args, **kwargs)
        print("Make a video took", time.time() - be)
        return out


def load_model(version=(MODEL_PREFIX + "melodyflow-t24-30secs")):
    global MODEL
    print("Loading model", version)
    if MODEL is None or MODEL.name != version:
        # Clear PyTorch CUDA cache and delete model
        del MODEL
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        MODEL = None  # in case loading would crash
        MODEL = MelodyFlow.get_pretrained(version)


def sanitize_filename(text, max_length=40):
    """Sanitize text to be used as a filename."""
    # Remove invalid filename characters
    sanitized = re.sub(r'[\\/*?:"<>|]', "", text)
    # Replace spaces and special chars with underscore
    sanitized = re.sub(r'[\s\-,;.]+', "_", sanitized)
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    # Remove trailing underscores
    sanitized = sanitized.strip("_")
    return sanitized


def _do_predictions(texts,
                    melodies,
                    solver,
                    steps,
                    target_flowstep,
                    regularize,
                    regularization_strength,
                    duration,
                    progress=False,
                    ):
    MODEL.set_generation_params(solver=solver,
                                steps=steps,
                                duration=duration,)
    MODEL.set_editing_params(solver=solver,
                             steps=steps,
                             target_flowstep=target_flowstep,
                             regularize=regularize,
                             lambda_kl=regularization_strength)
    print("new batch", len(texts), texts, [None if m is None else m for m in melodies])
    be = time.time()
    processed_melodies = []
    target_sr = 48000
    target_ac = 2
    for melody in melodies:
        if melody is None:
            processed_melodies.append(None)
        else:
            melody, sr = audio_read(melody)
            if melody.dim() == 2:
                melody = melody[None]
            if melody.shape[-1] > int(sr * MODEL.duration):
                melody = melody[..., :int(sr * MODEL.duration)]
            melody = convert_audio(melody, sr, target_sr, target_ac)
            melody = MODEL.encode_audio(melody.to(MODEL.device))
            processed_melodies.append(melody)

    try:
        if any(m is not None for m in processed_melodies):
            outputs = MODEL.edit(
                prompt_tokens=torch.cat(processed_melodies, dim=0).repeat(len(texts), 1, 1),
                descriptions=texts,
                src_descriptions=[""] * len(texts),
                progress=progress,
                return_tokens=False,
            )
        else:
            outputs = MODEL.generate(texts, progress=progress, return_tokens=False)
    except RuntimeError as e:
        raise gr.Error("Error while generating " + e.args[0])
    outputs = outputs.detach().cpu().float()
    out_wavs = []
    
    # Store all generated files with sanitized prompt-based filenames
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    for i, (output, text) in enumerate(zip(outputs, texts * (len(outputs) // len(texts)))):
        # Create a unique identifier
        short_uuid = str(uuid.uuid4())[:8]
        
        # Sanitize the prompt text for filename
        sanitized_text = sanitize_filename(text)
        
        # Create filename with sanitized prompt, timestamp, variation number and UUID
        variation = i % N_REPEATS + 1
        filename = f"{sanitized_text}_{timestamp}_var{variation}_{short_uuid}.wav"
        storage_path = STORAGE_DIR / filename
        
        # First save to a temporary file (as before)
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name, output, MODEL.sample_rate, strategy="loudness",
                loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
            out_wavs.append(file.name)
            file_cleaner.add(file.name)
            
            # Copy to storage directory with the new filename
            shutil.copy2(file.name, storage_path)
            print(f"Saved to storage: {storage_path}")
            
    print("batch finished", len(texts), time.time() - be)
    print("Tempfiles currently stored: ", len(file_cleaner.files))
    return out_wavs


@spaces.GPU(duration=30)
def predict(model, text,
                 solver, steps, target_flowstep,
                 regularize,
                 regularization_strength,
                 duration,
                 melody=None,
                 model_path=None,
                 progress=gr.Progress()):
    if melody is not None:
        if solver == MIDPOINT:
            steps = steps//2
        else:
            steps = steps//5

    global INTERRUPTING
    INTERRUPTING = False
    progress(0, desc="Loading model...")
    if model_path:
        model_path = model_path.strip()
        if not Path(model_path).exists():
            raise gr.Error(f"Model path {model_path} doesn't exist.")
        if not Path(model_path).is_dir():
            raise gr.Error(f"Model path {model_path} must be a folder containing "
                           "state_dict.bin and compression_state_dict_.bin.")
        model = model_path

    load_model(model)

    max_generated = 0

    def _progress(generated, to_generate):
        nonlocal max_generated
        max_generated = max(generated, max_generated)
        progress((min(max_generated, to_generate), to_generate))
        if INTERRUPTING:
            raise gr.Error("Interrupted.")
    MODEL.set_custom_progress_callback(_progress)

    wavs = _do_predictions(
        [text] * N_REPEATS, [melody],
        solver=solver,
        steps=steps,
        target_flowstep=target_flowstep,
        regularize=regularize,
        regularization_strength=regularization_strength,
        duration=duration,
        progress=True,)

    outputs_ = [wav for wav in wavs]
    return tuple(outputs_)


def toggle_audio_src(choice):
    if choice == "mic":
        return gr.update(sources=["microphone", "upload"], value=None, label="Microphone")
    else:
        return gr.update(sources=["upload", "microphone"], value=None, label="File")


def toggle_melody(melody):
    if melody is None:
        return gr.update(value=MIDPOINT)
    else:
        return gr.update(value=EULER)


def toggle_solver(solver, melody):
    if melody is None:
        if solver == MIDPOINT:
            return gr.update(value=64.0, minimum=2, maximum=128.0, step=2.0), gr.update(interactive=False, value=1.0), gr.update(interactive=False, value=False), gr.update(interactive=False, value=0.0), gr.update(interactive=True, value=30.0)
        else:
            return gr.update(value=64.0, minimum=1, maximum=128.0, step=1.0), gr.update(interactive=False, value=1.0), gr.update(interactive=False, value=False), gr.update(interactive=False, value=0.0), gr.update(interactive=True, value=30.0)
    else:
        if solver == MIDPOINT:
            return gr.update(value=128, minimum=4.0, maximum=256.0, step=4.0), gr.update(interactive=True, value=0.0), gr.update(interactive=False, value=False), gr.update(interactive=False, value=0.0), gr.update(interactive=False, value=0.0)
        else:
            return gr.update(value=125, minimum=5.0, maximum=250.0, step=5.0), gr.update(interactive=True, value=0.0), gr.update(interactive=True, value=True), gr.update(interactive=True, value=0.2), gr.update(interactive=False, value=0.0)

def ui_local(launch_kwargs):
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # MelodyFlow
            This is your private demo for [MelodyFlow](https://github.com/facebookresearch/audiocraft),
            A fast text-guided music generation and editing model based on a single-stage flow matching DiT
            presented at: ["High Fidelity Text-Guided Music Generation and Editing via Single-Stage Flow Matching"] (https://huggingface.co/papers/2407.03648)
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Input Text", interactive=True)
                    melody = gr.Audio(sources=["upload", "microphone"], type="filepath", label="File or Microphone",
                                      interactive=True, elem_id="melody-input", min_length=1)
                with gr.Row():
                    submit = gr.Button("Submit")
                    # Adapted from https://github.com/rkfg/audiocraft/blob/long/app.py, MIT license.
                    _ = gr.Button("Interrupt").click(fn=interrupt, queue=False)
                with gr.Row():
                    model = gr.Radio([(MODEL_PREFIX + "melodyflow-t24-30secs")],
                                     label="Model", value=(MODEL_PREFIX + "melodyflow-t24-30secs"), interactive=True)
                    model_path = gr.Text(label="Model Path (custom models)")
                with gr.Row():
                    solver = gr.Radio([EULER, MIDPOINT],
                                      label="ODE Solver", value=MIDPOINT, interactive=True)
                    steps = gr.Slider(label="Inference steps", minimum=2.0, maximum=128.0,
                                      step=2.0, value=128.0, interactive=True)
                    duration = gr.Slider(label="Duration", minimum=1.0, maximum=30.0, value=30.0, interactive=True)
                with gr.Row():
                    target_flowstep = gr.Slider(label="Target Flow step", minimum=0.0,
                                                maximum=1.0, value=0.0, interactive=False)
                    regularize = gr.Checkbox(label="Regularize", value=False, interactive=False)
                    regularization_strength = gr.Slider(
                        label="Regularization Strength", minimum=0.0, maximum=1.0, value=0.2, interactive=False)
            with gr.Column():
                audio_outputs = [
                    gr.Audio(label=f"Generated Audio - variation {i+1}", type='filepath', show_download_button=True, show_share_button=False) for i in range(N_REPEATS)]
        submit.click(fn=predict,
                     inputs=[model, text,
                             solver,
                             steps,
                             target_flowstep,
                             regularize,
                             regularization_strength,
                             duration,
                             melody,
                             model_path,],
                     outputs=[o for o in audio_outputs])
        melody.change(toggle_melody, melody, [solver])
        solver.change(toggle_solver, [solver, melody], [steps, target_flowstep,
                      regularize, regularization_strength, duration])
        gr.Examples(
            fn=predict,
            examples=[
                [
                    (MODEL_PREFIX + "melodyflow-t24-30secs"),
                    "80s electronic track with melodic synthesizers, catchy beat and groovy bass.",
                    MIDPOINT,
                    64,
                    1.0,
                    False,
                    0.0,
                    30.0,
                    None,
                ],
                [
                    (MODEL_PREFIX + "melodyflow-t24-30secs"),
                    "A cheerful country song with acoustic guitars accompanied by a nice piano melody.",
                    EULER,
                    125,
                    0.0,
                    True,
                    0.2,
                    -1.0,
                    "./assets/bolero_ravel.mp3",
                ],
            ],

            inputs=[model, text, solver, steps, target_flowstep,
                    regularize,
                    regularization_strength, duration, melody,],
            outputs=[audio_outputs],
            cache_examples=False,
        )

        gr.Markdown(
            """
            ### More details

            The model will generate a short music extract based on the description you provided.
            The model can generate or edit up to 30 seconds of audio in one pass.

            The model was trained with description from a stock music catalog, descriptions that will work best
            should include some level of details on the instruments present, along with some intended use case
            (e.g. adding "perfect for a commercial" can somehow help).

            You can optionally provide a reference audio from which the model will elaborate an edited version
            based on the text description, using MelodyFlow's regularized latent inversion.

            **WARNING:** Choosing long durations will take a longer time to generate.

            Available models are:
            1. facebook/melodyflow-t24-30secs (1B)

            See [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft/blob/main/docs/MELODYFLOW.md)
            for more details.
            """
        )

        interface.queue().launch(**launch_kwargs)

def ui_hf(launch_kwargs):
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # MelodyFlow
            This is the demo for [MelodyFlow](https://github.com/facebookresearch/audiocraft/blob/main/docs/MELODYFLOW.md),
            a fast text-guided music generation and editing model based on a single-stage flow matching DiT
            presented at: ["High Fidelity Text-Guided Music Generation and Editing via Single-Stage Flow Matching"](https://huggingface.co/papers/2407.03648).
            Use of this demo is subject to [Meta's AI Terms of Service](https://www.facebook.com/legal/ai-terms).
            <br/>
            <a href="https://huggingface.co/spaces/facebook/MelodyFlow?duplicate=true"
                style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank">
            <img style="margin-bottom: 0em;display: inline;margin-top: -.25em;"
                src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
            for longer sequences, more control and no queue.</p>
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Input Text", interactive=True)
                    melody = gr.Audio(sources=["upload", "microphone"], type="filepath", label="File or Microphone",
                                      interactive=True, elem_id="melody-input", min_length=1)
                with gr.Row():
                    submit = gr.Button("Submit")
                    # Adapted from https://github.com/rkfg/audiocraft/blob/long/app.py, MIT license.
                    _ = gr.Button("Interrupt").click(fn=interrupt, queue=False)
                with gr.Row():
                    model = gr.Radio([(MODEL_PREFIX + "melodyflow-t24-30secs")],
                                     label="Model", value=(MODEL_PREFIX + "melodyflow-t24-30secs"), interactive=True)
                with gr.Row():
                    solver = gr.Radio([EULER, MIDPOINT],
                                      label="ODE Solver", value=MIDPOINT, interactive=True)
                    steps = gr.Slider(label="Inference steps", minimum=2.0, maximum=128.0,
                                      step=2.0, value=128.0, interactive=True)
                    duration = gr.Slider(label="Duration", minimum=1.0, maximum=30.0, value=30.0, interactive=True)
                with gr.Row():
                    target_flowstep = gr.Slider(label="Target Flow step", minimum=0.0,
                                                maximum=1.0, value=0.0, interactive=False)
                    regularize = gr.Checkbox(label="Regularize", value=False, interactive=False)
                    regularization_strength = gr.Slider(
                        label="Regularization Strength", minimum=0.0, maximum=1.0, value=0.2, interactive=False)
            with gr.Column():
                audio_outputs = [
                    gr.Audio(label=f"Generated Audio - variation {i+1}", type='filepath', show_download_button=True, show_share_button=False) for i in range(N_REPEATS)]
        submit.click(fn=predict,
                     inputs=[model, text,
                             solver,
                             steps,
                             target_flowstep,
                             regularize,
                             regularization_strength,
                             duration,
                             melody,],
                     outputs=[o for o in audio_outputs])
        melody.change(toggle_melody, melody, [solver])
        solver.change(toggle_solver, [solver, melody], [steps, target_flowstep,
                      regularize, regularization_strength, duration])
        gr.Examples(
            fn=predict,
            examples=[
                [
                    (MODEL_PREFIX + "melodyflow-t24-30secs"),
                    "80s electronic track with melodic synthesizers, catchy beat and groovy bass.",
                    MIDPOINT,
                    64,
                    1.0,
                    False,
                    0.0,
                    30.0,
                    None,
                ],
                [
                    (MODEL_PREFIX + "melodyflow-t24-30secs"),
                    "A cheerful country song with acoustic guitars accompanied by a nice piano melody.",
                    EULER,
                    125,
                    0.0,
                    True,
                    0.2,
                    -1.0,
                    "./assets/bolero_ravel.mp3",
                ],
            ],

            inputs=[model, text, solver, steps, target_flowstep,
                    regularize,
                    regularization_strength, duration, melody,],
            outputs=[audio_outputs],
            cache_examples=False,
        )

        gr.Markdown("""
        ### More details

        The model will generate or edit up to 30 seconds of audio based on the description you provided.
        The model was trained with description from a stock music catalog, descriptions that will work best
        should include some level of details on the instruments present, along with some intended use case
        (e.g. adding "perfect for a commercial" can somehow help).

        You can optionally provide a reference audio from which the model will elaborate an edited version
        based on the text description, using MelodyFlow's regularized latent inversion.

        You can access more control (longer generation, more models etc.) by clicking
        the <a href="https://huggingface.co/spaces/facebook/MelodyFlow?duplicate=true"
                style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank">
            <img style="margin-bottom: 0em;display: inline;margin-top: -.25em;"
                src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
        (you will then need a paid GPU from HuggingFace).
        This gradio demo can also be run locally (best with GPU).

        See [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft/blob/main/docs/MELODYFLOW.md)
        for more details.
        """)

        interface.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='0.0.0.0' if 'SPACE_ID' in os.environ else '127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )

    args = parser.parse_args()

    launch_kwargs = {}
    launch_kwargs['server_name'] = args.listen

    if args.username and args.password:
        launch_kwargs['auth'] = (args.username, args.password)
    if args.server_port:
        launch_kwargs['server_port'] = args.server_port
    if args.inbrowser:
        launch_kwargs['inbrowser'] = args.inbrowser
    if args.share:
        launch_kwargs['share'] = args.share

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    # Show the interface
    if IS_HF_SPACE:
        ui_hf(launch_kwargs)
    else:
        ui_local(launch_kwargs)
