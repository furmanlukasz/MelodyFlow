# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
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
import torch
from audiocraft.data.audio import audio_read, audio_write
from audiocraft.data.audio_utils import convert_audio
from audiocraft.models import MelodyFlow

MODEL = None  # Last used model
MODEL_TYPE = None  # Last loaded model type (either "pretrained" or "custom")
MODEL_PREFIX = os.environ.get('MODEL_PREFIX', 'facebook/')
MAX_BATCH_SIZE = 12
N_REPEATS = 4
INTERRUPTING = False
MBD = None
# We have to wrap subprocess call to clean a bit the log when using gr.make_waveform
_old_call = sp.call

EULER = "euler"
MIDPOINT = "midpoint"

# Default path for custom models - use the specific path
DEFAULT_MODEL_PATH = "/workspace/MelodyFlow/melodyflow_finetuned_export_partial"

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


def load_model(version, model_path=None, force_reload=False):
    """Load MelodyFlow model, either pretrained or custom.
    
    Args:
        version: Base model name or "custom" for custom model
        model_path: Path to custom model directory
        force_reload: Whether to force reloading the model even if it's already loaded
    """
    global MODEL, MODEL_TYPE
    
    # Check if we already have the right model loaded
    if not force_reload and MODEL is not None:
        if (version == "custom" and MODEL_TYPE == "custom") or (version != "custom" and MODEL_TYPE == "pretrained" and MODEL.name == version):
            print(f"Model already loaded, skipping reload")
            return
    
    # Handle custom model case
    if version == "custom":
        if not model_path or not Path(model_path).exists():
            raise gr.Error(f"Custom model selected but valid model path not provided. Please enter a path to a directory containing state_dict.bin.")
        
        # Loading custom model from model_path
        print(f"Loading custom model from {model_path}")
        try:
            # Determine base model to use for configuration
            base_model = f"{MODEL_PREFIX}melodyflow-t24-30secs"
            
            # Clear existing model if any
            if MODEL is not None:
                del MODEL
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                MODEL = None
            
            # Get device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load the original model to get configuration
            original_model = MelodyFlow.get_pretrained(base_model, device=device)
            
            # Convert model_path to Path object
            model_path = Path(model_path)
            
            # Load the compression model
            compression_model = original_model.compression_model
            
            # Load our fine-tuned model state dict
            flow_state_dict_path = model_path / "state_dict.bin"
            if not flow_state_dict_path.exists():
                raise ValueError(f"Could not find state_dict.bin in {model_path}")
                
            flow_state_dict = torch.load(flow_state_dict_path, map_location=device)
            
            # Load the model configuration if available
            model_config_path = model_path / "model_config.bin"
            model_config = torch.load(model_config_path) if model_config_path.exists() else {}
            
            # Create a custom model with your fine-tuned weights
            custom_model = MelodyFlow(
                name="finetuned-melodyflow",
                compression_model=compression_model,
                lm=original_model.lm  # Start with original lm structure
            )
            
            # Load your fine-tuned weights
            custom_model.lm.load_state_dict(flow_state_dict)
            
            # Apply saved configuration if available
            if 'xp' in model_config:
                custom_model.lm.xp = model_config['xp']
                
            MODEL = custom_model
            MODEL_TYPE = "custom"
            print(f"Successfully loaded custom model from {model_path}")
            return
            
        except Exception as e:
            print(f"Error loading custom model: {e}")
            import traceback
            traceback.print_exc()
            raise gr.Error(f"Failed to load custom model: {str(e)}")
    
    # Loading standard pretrained model
    else:
        print(f"Loading pretrained model {version}")
        if MODEL is None or MODEL.name != version:
            # Clear PyTorch CUDA cache and delete model
            if MODEL is not None:
                del MODEL
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                MODEL = None  # in case loading would crash
            MODEL = MelodyFlow.get_pretrained(version)
            MODEL_TYPE = "pretrained"


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
    
    # Validate input parameters
    if model == "custom":
        if not model_path:
            raise gr.Error("Custom model selected but no model path provided. Please enter the path to your model directory.")
        if not Path(model_path).exists():
            raise gr.Error(f"Model path does not exist: {model_path}")
        if not Path(model_path).is_dir():
            raise gr.Error(f"Model path must be a directory: {model_path}")
        if not (Path(model_path) / "state_dict.bin").exists():
            raise gr.Error(f"state_dict.bin not found in {model_path}. Please ensure this is a valid MelodyFlow model directory.")
    
    # Handle model loading - only reload if needed
    load_model(model, model_path, force_reload=False)

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


def update_model_path_visibility(model_choice):
    """Update model path input visibility based on model choice"""
    if model_choice == "custom":
        return gr.update(visible=True, interactive=True, 
                        value=DEFAULT_MODEL_PATH,
                        placeholder="Enter path to model directory containing state_dict.bin")
    else:
        return gr.update(visible=False, interactive=False, value=None)


def ui_custom(launch_kwargs):
    # Preload the custom model at application startup
    try:
        print(f"Preloading custom model from {DEFAULT_MODEL_PATH}...")
        load_model("custom", DEFAULT_MODEL_PATH, force_reload=True)
        print("Model preloaded successfully!")
    except Exception as e:
        print(f"Warning: Failed to preload model: {e}")
        import traceback
        traceback.print_exc()
    
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # MelodyFlow Custom Model Interface
            This interface loads your fine-tuned MelodyFlow model from {}.
            """.format(DEFAULT_MODEL_PATH)
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Input Text", interactive=True, value="Energetic electronic music with a catchy melody")
                    melody = gr.Audio(sources=["upload", "microphone"], type="filepath", label="File or Microphone",
                                      interactive=True, elem_id="melody-input", min_length=1)
                with gr.Row():
                    model = gr.Radio(["custom", (MODEL_PREFIX + "melodyflow-t24-30secs")],
                                     label="Model", value="custom", interactive=True)
                    model_path = gr.Text(label="Custom Model Path", 
                                         value=DEFAULT_MODEL_PATH,
                                         placeholder="Enter path to model directory containing state_dict.bin",
                                         visible=True, interactive=True)
                with gr.Row():
                    submit = gr.Button("Submit")
                    # Adapted from https://github.com/rkfg/audiocraft/blob/long/app.py, MIT license.
                    _ = gr.Button("Interrupt").click(fn=interrupt, queue=False)
                with gr.Row():
                    solver = gr.Radio([EULER, MIDPOINT],
                                      label="ODE Solver", value=MIDPOINT, interactive=True)
                    steps = gr.Slider(label="Inference steps", minimum=2.0, maximum=128.0,
                                      step=2.0, value=64.0, interactive=True)
                    duration = gr.Slider(label="Duration", minimum=1.0, maximum=30.0, value=10.0, interactive=True)
                with gr.Row():
                    target_flowstep = gr.Slider(label="Target Flow step", minimum=0.0,
                                                maximum=1.0, value=0.0, interactive=False)
                    regularize = gr.Checkbox(label="Regularize", value=False, interactive=False)
                    regularization_strength = gr.Slider(
                        label="Regularization Strength", minimum=0.0, maximum=1.0, value=0.2, interactive=False)
            with gr.Column():
                audio_outputs = [
                    gr.Audio(label=f"Generated Audio - variation {i+1}", type='filepath', show_download_button=True, show_share_button=False) for i in range(N_REPEATS)]
        
        # Connect events
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
        model.change(update_model_path_visibility, [model], [model_path])
        
        # Add examples
        gr.Examples(
            fn=predict,
            examples=[
                [
                    "custom",
                    "A futuristic ambient soundscape with ethereal pads and gentle rhythms.",
                    EULER,
                    125,
                    0.0,
                    True,
                    0.2,
                    30.0,
                    None,
                    DEFAULT_MODEL_PATH
                ],
                [
                    "custom",
                    "An energetic electronic dance track with a driving beat and catchy synthesizer melodies.",
                    MIDPOINT,
                    64,
                    1.0,
                    False,
                    0.0,
                    30.0,
                    None,
                    DEFAULT_MODEL_PATH
                ],
                [
                    (MODEL_PREFIX + "melodyflow-t24-30secs"),
                    "A cheerful country song with acoustic guitars accompanied by a nice piano melody.",
                    EULER,
                    125,
                    0.0,
                    True,
                    0.2,
                    30.0,
                    "./assets/bolero_ravel.mp3",
                    None
                ],
            ],
            inputs=[model, text, solver, steps, target_flowstep,
                    regularize, regularization_strength, duration, melody, model_path],
            outputs=[audio_outputs],
            cache_examples=False,
        )

        gr.Markdown(
            """
            ### Using Custom Models
            
            To use a custom fine-tuned model:
            1. Select "custom" from the Model dropdown
            2. Enter the path to your model directory
               - This should be a directory containing `state_dict.bin` (fine-tuned weights)
               - Optionally, it can include `model_config.bin` for additional configuration
            
            ### Generation Settings
            
            - **ODE Solver**: Choose between Euler and Midpoint solvers
            - **Inference steps**: Higher values provide better quality at the cost of generation time
            - **Duration**: Length of generated audio in seconds
            - **Target Flow step**: Control the divergence from the original when editing
            - **Regularize**: Apply regularization when editing
            - **Regularization Strength**: Control the strength of regularization
            
            ### Reference Audio
            
            You can optionally provide a reference audio from which the model will elaborate an edited version
            based on the text description, using MelodyFlow's regularized latent inversion.
            
            All generated audio is saved to the `generated_melodies` folder with filenames derived from the prompt.
            """
        )

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

    # Launch the custom interface
    ui_custom(launch_kwargs) 