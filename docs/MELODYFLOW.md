# MelodyFlow: High Fidelity Text-Guided Music Editing via Single-Stage Flow Matching

AudioCraft provides the code and models for MelodyFlow, [High Fidelity Text-Guided Music Editing via Single-Stage Flow Matching][arxiv].

MelodyFlow is a text-guided music generation and editing model capable of generating high-quality stereo samples conditioned on text descriptions.
It is a Flow Matching Diffusion Transformer trained over a 48 kHz stereo (resp. 32 kHz mono) quantizer-free EnCodec tokenizer sampled at 25 Hz (resp. 20 Hz).
Unlike prior work on Flow Matching for music generation such as [MusicFlow: Cascaded Flow Matching for Text Guided Music Generation](https://openreview.net/forum?id=kOczKjmYum), 
MelodyFlow doesn't require model cascading, which makes it very convenient for music editing.

Check out our [sample page][melodyflow_samples] or test the available demo!

We use 16K hours of licensed music to train MelodyFlow. Specifically, we rely on an internal dataset
of 10K high-quality music tracks, and on the ShutterStock and Pond5 music data.


## Model Card

See [the model card](../model_cards/MELODFYFLOW_MODEL_CARD.md).


## Installation

Please follow the AudioCraft installation instructions from the [README](../README.md).

AudioCraft requires a GPU with at least 16 GB of memory for running inference with the medium-sized models (~1.5B parameters).

## Usage

We currently offer two ways to interact with MAGNeT:
1. You can use the gradio demo locally by running [`python -m demos.melodyflow_app --share`](../demos/melodyflow_app.py).
2. You can play with MelodyFlow by running the jupyter notebook at [`demos/melodyflow_demo.ipynb`](../demos/melodyflow_demo.ipynb) locally (also works on CPU).

## API

We provide a simple API and 1 pre-trained model:
- `facebook/melodyflow-t24-30secs`: 1B model, text to music, generates 30-second samples - [🤗 Hub](https://huggingface.co/facebook/melodyflow-t24-30secs)

See after a quick example for using the API.

```python
import torchaudio
from audiocraft.models import MelodyFlow
from audiocraft.data.audio import audio_write

model = MelodyFlow.get_pretrained('facebook/melodyflow-t24-30secs')
descriptions = ['disco beat', 'energetic EDM', 'funky groove']
wav = model.generate(descriptions)  # generates 3 samples.

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
```

## Training

Coming later...

## Citation
```
@misc{lan2024high,
      title={High fidelity text-guided music generation and editing via single-stage flow matching}, 
      author={Le Lan, Gael and Shi, Bowen and Ni, Zhaoheng and Srinivasan, Sidd and Kumar, Anurag and Ellis, Brian and Kant, David and Nagaraja, Varun and Chang, Ernie and Hsu, Wei-Ning and others},
      year={2024},
      eprint={2407.03648},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

## License

See license information in the [model card](../model_cards/MELODFYFLOW_MODEL_CARD.md).

[arxiv]: https://arxiv.org/pdf/2407.03648
[magnet_samples]: https://melodyflow.github.io/
