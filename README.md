# 9jaLingo TTS — African Speech Synthesis

**9jaLingo TTS** is a multilingual Text-to-Speech system trained from scratch for Nigerian and African languages. It supports **Hausa, Igbo, Yoruba, and Nigerian Pidgin**, with built-in **voice cloning** capability. The model was built by training the full Spark-TTS architecture on African language data using an NVIDIA A100 80GB GPU over 6 days.

> **Built on Spark-TTS** — This model is trained from scratch using the architecture from [Spark-TTS (arXiv:2503.01710)](https://arxiv.org/abs/2503.01710) by SparkAudio. Full credit and citation below. See their repo: [github.com/sparkaudio/spark-tts](https://github.com/sparkaudio/spark-tts)

> **License** — MIT with attribution. Free to use, modify, and distribute for any purpose. You must give credit to 9jaLingo and to the Spark-TTS authors. See [LICENSE](./LICENSE).

> This is a community release. If you use this model, please consider contributing data, fine-tunes, or improvements back to the African AI community.

---

## Supported Languages

| Language | Code | Speakers (cached) |
|---|---|---|
| Hausa | `hau` | 64 |
| Igbo | `ig` | 88 |
| Yoruba | `yor` | 81 |
| Nigerian Pidgin | `pcm` | 92 |
| **Total** | | **325** |

---

## How It Works — Architecture

9jaLingo TTS is based on the **Spark-TTS** architecture ([paper: arXiv 2503.01710](https://arxiv.org/abs/2503.01710)), which uses a two-stage pipeline to convert text into speech:

### Stage 1 — BiCodec Audio Tokenizer (4 days of training)

BiCodec is a neural audio codec that compresses speech into two types of discrete tokens:

- **Global tokens (32 tokens)** — capture the speaker's voice identity: timbre, accent, pitch style. These are what make the model speak in a particular person's voice.
- **Semantic tokens** — capture the content and prosody of the speech (what is said and how it is paced).

BiCodec can also run in reverse: given global tokens and semantic tokens, it reconstructs raw waveform audio at 16kHz.

The BiCodec encoder uses:
- A Wav2Vec2 SSL model (`wav2vec2-large-xlsr-53`) as the acoustic feature extractor
- A VQ (vector quantization) based quantizer for semantic tokens (codebook size: 8,192)
- A Finite Scalar Quantizer (FSQ) for global/speaker tokens
- A Vocos-style decoder to reconstruct waveforms

### Stage 2 — Language Model (2 days of training)

The LLM is an autoregressive transformer that takes text + speaker global tokens as input and predicts semantic speech tokens one by one. It has a vocabulary of **165,115 tokens** — standard text tokens plus special tokens for every possible BiCodec global and semantic code.

**Prompt format during inference:**
```
<|task_tts|><|start_content|>{text}<|end_content|>
<|start_global_token|>{speaker_global_tokens}<|end_global_token|>
```

The LLM generates `bicodec_semantic_N` tokens, which are then fed to the BiCodec decoder to produce the final audio.

### Two Inference Modes

| Mode | Input | How it works |
|---|---|---|
| **Standard TTS** | Speaker ID + text | Uses pre-cached global tokens (no audio upload needed) |
| **Voice Cloning** | Reference `.wav` + text | BiCodec encodes the reference to extract global tokens on the fly |

---

## Training Details

| Component | Duration | Hardware |
|---|---|---|
| BiCodec | 4 days | 1× NVIDIA A100 80GB |
| LLM | 2 days | 1× NVIDIA A100 80GB |
| **Total** | **6 days / 144 A100-GPU-hours** | |

- LLM checkpoint: **25,000 steps**
- Training was done completely from scratch on Nigerian language data
- The implementation follows the Spark-TTS paper and open-source code

---

## What Was Released

| Artifact | Status | Notes |
|---|---|---|
| BiCodec model weights | ✅ Released | `BiCodec/model.safetensors` |
| LLM model weights | ✅ Released | `LLM/` (HuggingFace format) |
| Speaker global tokens cache | ✅ Released | 325 speakers across 4 languages |
| Inference notebook (Colab) | ✅ Released | `inference_spark_tts.ipynb` |
| Quick test notebook (Colab) | ✅ Released | `colab_test.ipynb` |
| Inference Python script | ✅ Released | `src/inference.py` |
| BiCodec training notebook | 🔜 Coming soon | |
| LLM training notebook | 🔜 Coming soon | |

---

## Model on HuggingFace

The model weights are hosted on HuggingFace:

```
NaijaLingo/9jaLingo-TTS-African-ckpt-25k
```

Directory structure in the HuggingFace repo:

```
9jaLingo-TTS-African-ckpt-25k/
├── BiCodec/
│   ├── model.safetensors        # BiCodec audio tokenizer weights
│   └── config.yaml
├── LLM/                         # LLM weights (HuggingFace format)
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── model.safetensors
│   └── generation_config.json
├── wav2vec2-large-xlsr-53/      # SSL feature extractor
├── config.yaml                  # Top-level BiCodec config
└── speaker_global_tokens.json   # Pre-cached speaker identities (325 speakers)
```

---

## Installation

### Requirements

- Python 3.9+
- CUDA GPU recommended (CPU inference is very slow)
- ~10 GB VRAM for comfortable inference (A100, V100, RTX 3090+)

### Step 1 — Clone SparkVox (BiCodec code)

```bash
git clone https://github.com/SparkAudio/SparkVox.git
cd SparkVox
```

Apply the PyTorch 2.6+ compatibility patch:

```python
# In SparkVox/sparkvox/models/codec/BiCodec/modules/generator.py
# Change:
torch.load(ckpt_path, map_location="cpu")
# To:
torch.load(ckpt_path, map_location="cpu", weights_only=False)
```

### Step 2 — Install Dependencies

```bash
pip install lightning pytorch-lightning hydra-core omegaconf
pip install librosa soundfile torchaudio einops
pip install transformers huggingface_hub
pip install vector-quantize-pytorch descript-audiotools
pip install protobuf==3.20.3 safetensors accelerate sentencepiece
```

### Step 3 — Download Model from HuggingFace

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="NaijaLingo/9jaLingo-TTS-African-ckpt-25k",
    local_dir="./Spark-TTS-African",
    local_dir_use_symlinks=False
)
```

---

## Usage

### Quick Start (Python script)

```bash
cd SparkVox   # must run from SparkVox directory
python /path/to/9jalingo-Spark-tts/src/inference.py \
    --model_dir ./Spark-TTS-African \
    --mode standard \
    --speaker_id hau_f_915 \
    --text "Rayuwa tafiya ce mai cike da darussa, farin ciki da kalubale." \
    --output output.wav
```

### Voice Cloning

```bash
python /path/to/9jalingo-Spark-tts/src/inference.py \
    --model_dir ./Spark-TTS-African \
    --mode clone \
    --reference_audio my_voice.wav \
    --text "Text to speak in the cloned voice." \
    --output cloned_output.wav
```

### List Available Speakers

```bash
python /path/to/9jalingo-Spark-tts/src/inference.py \
    --model_dir ./Spark-TTS-African \
    --list_speakers \
    --language hau \
    --gender female
```

### Python API

```python
import sys
sys.path.insert(0, "/path/to/SparkVox")

from src.inference import NaijaLingoTTS

tts = NaijaLingoTTS(model_dir="./Spark-TTS-African", sparkvox_dir="./SparkVox")

# Standard TTS — pick a speaker from the cache
wav = tts.speak("Ndụ bụ njem nke jupụtara na ihe nkuzi.", speaker_id="IECT1F11")
tts.save(wav, "output.wav")

# List speakers by language
tts.list_speakers(language="ig", gender="female")

# Voice Cloning — provide reference audio
wav = tts.clone("Any text you want spoken.", reference_audio="reference.wav")
tts.save(wav, "cloned.wav")
```

### Google Colab

**Quick test** — clone repo, load model, run standard TTS + voice clone in minutes:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/9jaLingo/9jalingo-Spark-tts/blob/main/colab_test.ipynb)

**Full inference notebook** — all speakers, batch generation, detailed controls:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/9jaLingo/9jalingo-Spark-tts/blob/main/inference_spark_tts.ipynb)

---

## Example Speakers

| Speaker ID | Language | Gender | Samples |
|---|---|---|---|
| `hau_f_915` | Hausa | Female | 4,278 |
| `hau_m_954` | Hausa | Male | 3,200 |
| `IECT1F11` | Igbo | Female | 566 |
| `YBUT2M21` | Yoruba | Male | 599 |
| `PP1F3` | Pidgin | Female | 600 |

Use `list_speakers()` to browse all 325 speakers, with filters for language and gender.

---

## Generation Parameters

| Parameter | Default | Description |
|---|---|---|
| `temperature` | `0.8` | Controls variation. Lower (0.5) = more stable, higher (1.0) = more expressive |
| `top_k` | `50` | Top-k sampling |
| `top_p` | `0.95` | Nucleus sampling threshold |
| `max_new_tokens` | `3000` | Maximum semantic tokens to generate |

---

## Limitations

- The model was trained on ~25k LLM steps. More training steps will improve naturalness and robustness.
- Very long texts (>30 words) may occasionally truncate. Split into sentences for best results.
- Output audio is at 16kHz mono.
- The model has not been evaluated on formal benchmarks — this is a research and community release.

---

## Acknowledgements

This model would not exist without the foundational work from the **Spark-TTS** team at SparkAudio:

> **Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens**
> arXiv:2503.01710 — [https://arxiv.org/abs/2503.01710](https://arxiv.org/abs/2503.01710)
>
> GitHub: [https://github.com/sparkaudio/spark-tts](https://github.com/sparkaudio/spark-tts)
>
> BiCodec / SparkVox: [https://github.com/SparkAudio/SparkVox](https://github.com/SparkAudio/SparkVox)

The architecture, training framework, and BiCodec tokenizer design are all from the Spark-TTS paper. This project trained the full system from scratch on Nigerian language data to bring high-quality TTS to African languages.

---

## About 9jaLingo

**9jaLingo** is a platform building language technology for Nigerian and African languages — speech recognition, TTS, translation, and NLP tools that center African voices.

This model is one piece of that mission: giving African languages a voice in AI.

- Platform: [9jalingo.com](https://9jalingo.com)
- HuggingFace: [huggingface.co/NaijaLingo](https://huggingface.co/NaijaLingo)
- GitHub: [github.com/NaijaLingo](https://github.com/NaijaLingo)

---

## Citation

If you use this model in your research or products, please cite both the original Spark-TTS paper and this repository:

```bibtex
@article{sparktts2025,
  title={Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens},
  author={SparkAudio Team},
  journal={arXiv preprint arXiv:2503.01710},
  year={2025}
}

@misc{9jalingo_tts_2025,
  title={9jaLingo TTS — African Speech Synthesis},
  author={Chukwuemeka Okolie},
  year={2025},
  publisher={9jaLingo},
  howpublished={\url{https://github.com/NaijaLingo/9jalingo-Spark-tts}}
}
```

---

## License

Model weights and inference code are released under the **Apache 2.0 License**.

The SparkVox and Spark-TTS codebases are subject to their own licenses — see the respective repositories.

---

## Contributing

Contributions are welcome:

- **More data**: Audio recordings in Hausa, Igbo, Yoruba, Pidgin, or other Nigerian/African languages
- **Fine-tuning**: Improvements on top of the released checkpoint
- **New languages**: Extending to Efik, Tiv, Fulfulde, Amharic, Swahili, and others
- **Bug reports and feedback**: Open a GitHub issue

Let's build African AI together.
