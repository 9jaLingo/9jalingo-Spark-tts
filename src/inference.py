"""
9jaLingo TTS — Inference Script
================================
Multilingual TTS for Nigerian/African languages (Hausa, Igbo, Yoruba, Pidgin).

LICENSE: MIT with attribution. Free to use for any purpose.
You must give credit to:
  - 9jaLingo TTS — Chukwuemeka Okolie / 9jaLingo (https://9jalingo.com)
  - Spark-TTS (arXiv:2503.01710) — https://github.com/sparkaudio/spark-tts

Built on the Spark-TTS architecture (arXiv:2503.01710).

IMPORTANT: This script must be run from the SparkVox directory, OR you must
pass --sparkvox_dir to point at where you cloned SparkVox.

  git clone https://github.com/SparkAudio/SparkVox.git
  cd SparkVox
  python /path/to/9jalingo-Spark-tts/src/inference.py --help

Two modes:
  standard  : pick a speaker_id from the cache, no audio upload needed
  clone     : provide a reference .wav, model clones that voice

Examples:
  # List all Hausa female speakers
  python src/inference.py --model_dir ./Spark-TTS-African --list_speakers --language hau --gender female

  # Generate speech in Hausa
  python src/inference.py --model_dir ./Spark-TTS-African \\
      --mode standard --speaker_id hau_f_915 \\
      --text "Rayuwa tafiya ce mai cike da darussa." \\
      --output output.wav

  # Clone a voice
  python src/inference.py --model_dir ./Spark-TTS-African \\
      --mode clone --reference_audio my_voice.wav \\
      --text "Speak this in my voice." \\
      --output cloned.wav
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import soundfile as sf
import torch


LANG_NAMES = {"ig": "Igbo", "yor": "Yoruba", "hau": "Hausa", "pcm": "Pidgin"}


# ---------------------------------------------------------------------------
# Compatibility patch for PyTorch >= 2.6 + safetensors BiCodec checkpoint
# ---------------------------------------------------------------------------

def _apply_torch_patch(sparkvox_dir: str):
    gen_path = os.path.join(
        sparkvox_dir,
        "sparkvox/models/codec/BiCodec/modules/generator.py",
    )
    if not os.path.exists(gen_path):
        print(f"Warning: generator.py not found at {gen_path} — skipping patch")
        return
    with open(gen_path, "r") as f:
        src = f.read()
    if "weights_only=False" not in src:
        src = src.replace(
            'torch.load(ckpt_path, map_location="cpu")',
            'torch.load(ckpt_path, map_location="cpu", weights_only=False)',
        )
        with open(gen_path, "w") as f:
            f.write(src)
        print("Patched generator.py for PyTorch 2.6+ compatibility")


def _build_torch_load_patch():
    """Monkeypatch torch.load so safetensors BiCodec checkpoints load correctly."""
    from safetensors.torch import load_file as _load_safetensors

    _original_torch_load = torch.load

    def _patched(f, *args, **kwargs):
        filepath = str(f if isinstance(f, (str, os.PathLike)) else "")
        if filepath.endswith(".safetensors"):
            flat = _load_safetensors(filepath)
            # Generator.load_from_checkpoint expects the model.generator. prefix
            prefixed = {f"model.generator.{k}": v for k, v in flat.items()}
            return {"state_dict": prefixed}
        kwargs["weights_only"] = False
        return _original_torch_load(f, *args, **kwargs)

    torch.load = _patched
    return _original_torch_load


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def normalize_audio(wav, target_db: float = -20.0) -> np.ndarray:
    if isinstance(wav, torch.Tensor):
        wav = wav.cpu().numpy()
    wav = wav.astype(np.float32).squeeze()
    wav -= np.mean(wav)
    rms = np.sqrt(np.mean(wav ** 2))
    if rms > 1e-6:
        target_rms = 10 ** (target_db / 20)
        wav = wav * (target_rms / rms)
    wav = np.tanh(wav)
    return wav


# ---------------------------------------------------------------------------
# Main TTS class
# ---------------------------------------------------------------------------

class NaijaLingoTTS:
    """
    9jaLingo TTS inference engine.

    Usage:
        tts = NaijaLingoTTS(model_dir="./Spark-TTS-African", sparkvox_dir="./SparkVox")

        # Standard TTS (cached speaker)
        wav = tts.speak("Ndụ bụ njem.", speaker_id="IECT1F11")
        tts.save(wav, "out.wav")

        # Voice cloning
        wav = tts.clone("Say this in my voice.", reference_audio="ref.wav")
        tts.save(wav, "cloned.wav")

        # List speakers
        tts.list_speakers(language="ig", gender="female")
    """

    def __init__(self, model_dir: str, sparkvox_dir: str = None, device: str = None):
        self.model_dir = os.path.abspath(model_dir)

        # Resolve SparkVox directory
        if sparkvox_dir is None:
            sparkvox_dir = os.path.join(os.getcwd(), "SparkVox")
            if not os.path.isdir(sparkvox_dir):
                sparkvox_dir = os.getcwd()  # assume already inside SparkVox
        self.sparkvox_dir = os.path.abspath(sparkvox_dir)

        if self.sparkvox_dir not in sys.path:
            sys.path.insert(0, self.sparkvox_dir)

        # GPU / CPU
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Device: {self.device}")
        print(f"Model : {self.model_dir}")
        print(f"SparkVox: {self.sparkvox_dir}")

        _apply_torch_patch(self.sparkvox_dir)
        self._audio_tok = None
        self._hf_tokenizer = None
        self._hf_model = None
        self._speaker_cache = None

        self._load_models()
        self._load_speaker_cache()

    # ------------------------------------------------------------------ load

    def _load_models(self):
        original_torch_load = _build_torch_load_patch()

        import hydra
        from sparkvox.utils.file import load_config

        bicodec_dir = os.path.join(self.model_dir, "BiCodec")
        wav2vec_dir = os.path.join(self.model_dir, "wav2vec2-large-xlsr-53")
        bicodec_ckpt = os.path.join(bicodec_dir, "model.safetensors")

        if not os.path.exists(bicodec_ckpt):
            raise FileNotFoundError(
                f"BiCodec weights not found: {bicodec_ckpt}\n"
                "Download the model from HuggingFace: 9jaLingo/9jaLingo-TTS-African-ckpt-25k"
            )

        # Read top-level config
        top_config = load_config(os.path.join(self.model_dir, "config.yaml"))
        sample_rate = top_config.get("sample_rate", 16000)
        self.sample_rate = sample_rate
        ref_dur = top_config.get("ref_segment_duration", 6)
        hop = top_config.get("latent_hop_length", 320)
        vol_norm = top_config.get("volume_normalize", True)

        # Write temp BiCodec config
        tmp_dir = os.path.join(self.sparkvox_dir, "local", "inference_config")
        os.makedirs(tmp_dir, exist_ok=True)
        bicodec_config_path = os.path.join(tmp_dir, "bicodec_config.yaml")
        _write_bicodec_config(
            bicodec_config_path, sample_rate, ref_dur, hop, vol_norm
        )

        audio_tok_yaml_path = os.path.join(tmp_dir, "audio_tokenizer.yaml")
        with open(audio_tok_yaml_path, "w") as f:
            f.write(
                f'audio_tokenizer:\n'
                f'  _target_: sparkvox.tools.tokenizer.audio_tokenizer.bicodec.audio_tokenizer.BiCodecTokenizer\n'
                f'  config_path: "{bicodec_config_path}"\n'
                f'  ckpt_path: "{bicodec_ckpt}"\n'
                f'  wav2vec_model: "{wav2vec_dir}"\n'
            )

        print("Loading BiCodec audio tokenizer...")
        cfg = load_config(audio_tok_yaml_path)
        self._audio_tok = hydra.utils.instantiate(
            cfg["audio_tokenizer"], device=self.device
        )

        from transformers import AutoModelForCausalLM, AutoTokenizer

        llm_dir = os.path.join(self.model_dir, "LLM")
        print(f"Loading LLM from {llm_dir}...")
        self._hf_tokenizer = AutoTokenizer.from_pretrained(llm_dir)
        self._hf_model = (
            AutoModelForCausalLM.from_pretrained(llm_dir).to(self.device)
        )
        self._hf_model.eval()
        print(f"LLM vocab size: {self._hf_model.config.vocab_size:,}")

        torch.load = original_torch_load
        print("Models loaded.")

    def _load_speaker_cache(self):
        cache_path = os.path.join(self.model_dir, "speaker_global_tokens.json")
        if not os.path.exists(cache_path):
            print(f"Warning: speaker cache not found at {cache_path}")
            self._speaker_cache = {}
            return
        with open(cache_path, "r") as f:
            self._speaker_cache = json.load(f)
        print(f"Loaded {len(self._speaker_cache)} speakers from cache")

    # --------------------------------------------------------------- standard

    @torch.no_grad()
    def speak(
        self,
        text: str,
        speaker_id: str,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        max_new_tokens: int = 3000,
    ) -> np.ndarray:
        """
        Generate speech using a cached speaker identity.

        Args:
            text:           Text to synthesise.
            speaker_id:     Key from speaker_global_tokens.json.
            temperature:    Sampling temperature. Lower = more stable.
            top_k:          Top-k sampling.
            top_p:          Nucleus sampling probability.
            max_new_tokens: Maximum semantic tokens to generate.

        Returns:
            NumPy float32 array, 16 kHz.
        """
        if speaker_id not in self._speaker_cache:
            available = list(self._speaker_cache.keys())[:5]
            raise ValueError(
                f"Speaker '{speaker_id}' not in cache.\n"
                f"Examples: {available}\nUse list_speakers() to browse."
            )

        info = self._speaker_cache[speaker_id]
        global_str = info["global_str"]
        global_ids = info["global_ids"]

        prompt = (
            f"<|task_tts|><|start_content|>{text}<|end_content|>"
            f"<|start_global_token|>{global_str}<|end_global_token|>"
        )

        model_inputs = self._hf_tokenizer([prompt], return_tensors="pt").to(self.device)
        generated_ids = self._hf_model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        new_ids = generated_ids[0][model_inputs.input_ids.shape[1]:]
        decoded = self._hf_tokenizer.decode(new_ids, skip_special_tokens=True)

        semantic_ids = [int(t) for t in re.findall(r"bicodec_semantic_(\d+)", decoded)]
        if not semantic_ids:
            print("Warning: no semantic tokens generated — try again or lower temperature")
            return np.zeros(self.sample_rate, dtype=np.float32)

        print(f"  Generated {len(semantic_ids)} semantic tokens")

        pred_semantic = torch.tensor(semantic_ids).long().unsqueeze(0).to(self.device)
        global_tensor = torch.tensor(global_ids).long().unsqueeze(0).unsqueeze(0).to(self.device)
        wav = self._audio_tok.detokenize(global_tensor, pred_semantic)
        return normalize_audio(wav)

    # ----------------------------------------------------------------- clone

    @torch.no_grad()
    def clone(
        self,
        text: str,
        reference_audio: str,
        prompt_text: str = None,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        max_new_tokens: int = 3000,
    ) -> np.ndarray:
        """
        Generate speech by cloning a voice from reference audio.

        Args:
            text:            Text to synthesise.
            reference_audio: Path to reference .wav (3–10 seconds recommended).
            prompt_text:     Transcript of the reference audio (optional).
                             Providing this enables cross-utterance cloning.
            temperature:     Sampling temperature.
            top_k:           Top-k sampling.
            top_p:           Nucleus sampling probability.
            max_new_tokens:  Maximum semantic tokens to generate.

        Returns:
            NumPy float32 array, 16 kHz.
        """
        global_tokens, semantic_tokens = self._audio_tok.tokenize(reference_audio)
        global_str = "".join(
            [f"<|bicodec_global_{int(i)}|>" for i in global_tokens.squeeze().flatten()]
        )

        if prompt_text:
            semantic_str = "".join(
                [f"<|bicodec_semantic_{int(i)}|>" for i in semantic_tokens.squeeze().flatten()]
            )
            prompt = (
                f"<|task_tts|><|start_content|>{prompt_text}{text}<|end_content|>"
                f"<|start_global_token|>{global_str}<|end_global_token|>"
                f"<|start_semantic_token|>{semantic_str}<|end_semantic_token|>"
            )
        else:
            prompt = (
                f"<|task_tts|><|start_content|>{text}<|end_content|>"
                f"<|start_global_token|>{global_str}<|end_global_token|>"
            )

        model_inputs = self._hf_tokenizer([prompt], return_tensors="pt").to(self.device)
        generated_ids = self._hf_model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        new_ids = generated_ids[0][model_inputs.input_ids.shape[1]:]
        decoded = self._hf_tokenizer.decode(new_ids, skip_special_tokens=True)

        semantic_ids = [int(t) for t in re.findall(r"bicodec_semantic_(\d+)", decoded)]
        if not semantic_ids:
            print("Warning: no semantic tokens generated — try again or lower temperature")
            return np.zeros(self.sample_rate, dtype=np.float32)

        print(f"  Generated {len(semantic_ids)} semantic tokens")

        pred_semantic = torch.tensor(semantic_ids).long().unsqueeze(0).to(self.device)
        wav = self._audio_tok.detokenize(global_tokens.to(self.device), pred_semantic)
        return normalize_audio(wav)

    # --------------------------------------------------------------- helpers

    def save(self, wav: np.ndarray, path: str):
        """Write a WAV file at 16 kHz."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        sf.write(path, wav, self.sample_rate)
        print(f"Saved: {path}")

    def list_speakers(
        self,
        language: str = None,
        gender: str = None,
        limit: int = 30,
    ) -> list:
        """
        Print and return matching speaker IDs.

        Args:
            language: 'hau', 'ig', 'yor', 'pcm', or None for all.
            gender:   'male', 'female', or None for all.
            limit:    Max rows to display.
        """
        rows = [
            (sid, info)
            for sid, info in self._speaker_cache.items()
            if (language is None or info.get("language") == language)
            and (gender is None or info.get("gender") == gender)
        ]

        sep = "-" * 56
        print(sep)
        print(f"{'Speaker ID':<22} {'Language':<10} {'Gender':<10} Samples")
        print(sep)
        for sid, info in rows[:limit]:
            lang = LANG_NAMES.get(info["language"], info["language"])
            print(
                f"{sid:<22} {lang:<10} {info['gender']:<10} "
                f"{info.get('num_samples', '?')}"
            )
        if len(rows) > limit:
            print(f"  ... and {len(rows) - limit} more (increase --limit to see more)")
        print(sep)
        print(f"Total: {len(rows)} speakers")
        return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# BiCodec YAML writer
# ---------------------------------------------------------------------------

def _write_bicodec_config(path, sample_rate, ref_dur, hop, vol_norm):
    content = f"""datasets:
  sample_rate: {sample_rate}
  sample_rate_for_ssl: {sample_rate}
  segment_duration: 2.4
  max_val_duration: 12
  latent_hop_length: {hop}
  ref_segment_duration: {ref_dur}
  additonal_duration_per_side: 1
  volume_normalize: {str(vol_norm).lower()}

model:
  generator:
    mel_params:
      sample_rate: {sample_rate}
      n_fft: 1024
      win_length: 640
      hop_length: {hop}
      mel_fmin: 10
      mel_fmax: null
      num_mels: 128
    encoder:
      _target_: sparkvox.models.codec.BiCodec.modules.feat_encoder.Encoder
      input_channels: 1024
      vocos_dim: 384
      vocos_intermediate_dim: 2048
      vocos_num_layers: 12
      out_channels: 1024
      sample_ratios: [1, 1]
    decoder:
      _target_: sparkvox.models.codec.base.modules.wave_generator_dac.Decoder
      input_channel: 1024
      channels: 1536
      rates: [8, 5, 4, 2]
      kernel_sizes: [16, 11, 8, 4]
    quantizer:
      _target_: sparkvox.models.codec.base.quantize.factorized_vector_quantize.FactorizedVectorQuantize
      input_dim: 1024
      codebook_size: 8192
      codebook_dim: 8
      commitment: 0.25
      codebook_loss_weight: 4.0
      use_l2_normlize: true
      threshold_ema_dead_code: 0.2
    speaker_encoder:
      _target_: sparkvox.models.codec.BiCodec.modules.speaker_encoder.SpeakerEncoder
      input_dim: 128
      out_dim: 1024
      latent_dim: 128
      token_num: 32
      fsq_levels: [4, 4, 4, 4, 4, 4]
      fsq_num_quantizers: 1
    prenet:
      _target_: sparkvox.models.codec.BiCodec.modules.feat_decoder.Decoder
      input_channels: 1024
      vocos_dim: 384
      vocos_intermediate_dim: 2048
      vocos_num_layers: 12
      out_channels: 1024
      condition_dim: 1024
      sample_ratios: [1, 1]
      use_tanh_at_final: false
    postnet:
      _target_: sparkvox.models.codec.BiCodec.modules.feat_decoder.Decoder
      input_channels: 1024
      vocos_dim: 384
      vocos_intermediate_dim: 2048
      vocos_num_layers: 6
      out_channels: 1024
      use_tanh_at_final: false
    d_vector_train_start: 1000
"""
    with open(path, "w") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="9jaLingo TTS — African multilingual speech synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model_dir",
        default="./Spark-TTS-African",
        help="Path to downloaded model directory (default: ./Spark-TTS-African)",
    )
    parser.add_argument(
        "--sparkvox_dir",
        default=None,
        help="Path to SparkVox repo clone. Defaults to ./SparkVox relative to cwd.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="'cuda' or 'cpu'. Auto-detected if not set.",
    )

    # List mode
    parser.add_argument(
        "--list_speakers",
        action="store_true",
        help="List available speakers and exit.",
    )
    parser.add_argument(
        "--language",
        default=None,
        choices=["hau", "ig", "yor", "pcm"],
        help="Filter speakers by language.",
    )
    parser.add_argument(
        "--gender",
        default=None,
        choices=["male", "female"],
        help="Filter speakers by gender.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Max speakers to display in list mode (default: 30).",
    )

    # Generation mode
    parser.add_argument(
        "--mode",
        choices=["standard", "clone"],
        default="standard",
        help="'standard' uses cached speaker; 'clone' uses reference audio.",
    )
    parser.add_argument("--text", default=None, help="Text to synthesise.")
    parser.add_argument(
        "--speaker_id",
        default=None,
        help="Speaker ID from cache (standard mode). Use --list_speakers to find IDs.",
    )
    parser.add_argument(
        "--reference_audio",
        default=None,
        help="Path to reference .wav file (clone mode).",
    )
    parser.add_argument(
        "--prompt_text",
        default=None,
        help="Transcript of reference audio (optional, clone mode only).",
    )
    parser.add_argument(
        "--output",
        default="output.wav",
        help="Output WAV file path (default: output.wav).",
    )

    # Sampling
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8). Lower = more stable.",
    )
    parser.add_argument("--top_k", type=int, default=50, help="Top-k (default: 50).")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (default: 0.95).")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=3000,
        help="Max semantic tokens to generate (default: 3000).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    tts = NaijaLingoTTS(
        model_dir=args.model_dir,
        sparkvox_dir=args.sparkvox_dir,
        device=args.device,
    )

    if args.list_speakers:
        tts.list_speakers(language=args.language, gender=args.gender, limit=args.limit)
        return

    if not args.text:
        print("Error: --text is required for generation. Use --list_speakers to browse speakers.")
        sys.exit(1)

    if args.mode == "standard":
        if not args.speaker_id:
            print("Error: --speaker_id is required in standard mode.")
            print("Use --list_speakers to browse available speakers.")
            sys.exit(1)
        info = tts._speaker_cache.get(args.speaker_id, {})
        lang = LANG_NAMES.get(info.get("language", "?"), info.get("language", "?"))
        print(f"Speaker : {args.speaker_id} | {lang} | {info.get('gender', '?')}")
        print(f"Text    : {args.text}")
        wav = tts.speak(
            text=args.text,
            speaker_id=args.speaker_id,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )

    elif args.mode == "clone":
        if not args.reference_audio:
            print("Error: --reference_audio is required in clone mode.")
            sys.exit(1)
        if not os.path.exists(args.reference_audio):
            print(f"Error: reference audio not found: {args.reference_audio}")
            sys.exit(1)
        print(f"Reference: {args.reference_audio}")
        print(f"Text     : {args.text}")
        wav = tts.clone(
            text=args.text,
            reference_audio=args.reference_audio,
            prompt_text=args.prompt_text or None,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )

    tts.save(wav, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
