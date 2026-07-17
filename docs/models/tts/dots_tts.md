# Dots TTS

Dots TTS is a multilingual voice-cloning model ported to MLX for Apple Silicon. The current `mlx-audio` integration supports local converted checkpoints and has been verified on split Hugging Face repos under `mlx-community`.

## Model Variants

| Model | Notes | HuggingFace |
|-------|-------|-------------|
| `mlx-community/dots-tts-mlx-int4` | Default quality/size tradeoff; 48 kHz output | [:octicons-link-external-16: Model Card](https://huggingface.co/mlx-community/dots-tts-mlx-int4) |
| `mlx-community/dots-tts-mlx-int8` | Larger footprint with int8 LLM quantization | [:octicons-link-external-16: Model Card](https://huggingface.co/mlx-community/dots-tts-mlx-int8) |
| `mlx-community/dots-tts-mlx-mf-int4` | Meanflow variant, int4 LLM quantization | [:octicons-link-external-16: Model Card](https://huggingface.co/mlx-community/dots-tts-mlx-mf-int4) |
| `mlx-community/dots-tts-mlx-mf-int8` | Meanflow variant, int8 LLM quantization | [:octicons-link-external-16: Model Card](https://huggingface.co/mlx-community/dots-tts-mlx-mf-int8) |

!!! note
    The loader expects the local checkpoint directory itself to contain `config.json`, `llm_config.json`, `core.safetensors`, `vocoder.safetensors`, `speaker.safetensors`, `latent_stats.npz`, and a `tokenizer/` directory.

## Features

- Multilingual TTS
- Zero-shot voice cloning
- 48 kHz waveform output
- Local-path friendly loading for converted checkpoints
- Flow-matching and meanflow checkpoint variants

## Usage

### Basic Generation

=== "CLI"

    ```bash
    mlx_audio.tts.generate \
        --model mlx-community/dots-tts-mlx-int4 \
        --text "Hello from Dots TTS on MLX." \
        --lang_code en
    ```

=== "Python"

    ```python
    from mlx_audio.tts.utils import load_model

    model = load_model("mlx-community/dots-tts-mlx-int4")
    result = next(model.generate(
        text="Hello from Dots TTS on MLX.",
        lang_code="en",
    ))

    audio = result.audio
    ```

### Voice Cloning

Dots cloning requires both the reference audio and its transcript.

=== "CLI"

    ```bash
    mlx_audio.tts.generate \
        --model mlx-community/dots-tts-mlx-int4 \
        --text "This should sound like the reference speaker." \
        --lang_code en \
        --ref_audio reference.wav \
        --ref_text "This is what the reference speaker says."
    ```

=== "Python"

    ```python
    from mlx_audio.tts.utils import load_model

    model = load_model("mlx-community/dots-tts-mlx-int4")
    result = next(model.generate(
        text="This should sound like the reference speaker.",
        ref_audio="reference.wav",
        ref_text="This is what the reference speaker says.",
        lang_code="en",
    ))

    audio = result.audio
    ```

!!! important
    Dots currently supports a single `ref_audio` / `ref_text` pair per request. Passing
    `ref_text` is required for cloning. Path-based `ref_audio` currently expects a WAV
    file; for non-WAV assets, decode them externally and pass a predecoded mono array.

## Languages

The verified `mlx-community/dots-tts-mlx-*` checkpoints advertise support for 24 languages:
Arabic, Cantonese, Chinese, Czech, Dutch, English, Finnish, French, German, Greek, Hindi, Indonesian, Italian, Japanese, Korean, Polish, Portuguese, Romanian, Russian, Spanish, Thai, Turkish, Ukrainian, and Vietnamese.

## Notes

- The wrapper preserves the raw reference audio path so the Dots runtime can handle its own prompt-audio loading and resampling.
- The current implementation normalizes `lang_code="en"` style inputs to uppercase runtime language tags.
- On memory-constrained Apple Silicon machines, set `MLX_AUDIO_DOTS_MEMORY_LIMIT_GB` to a conservative value before loading the model.
- Multi-variant roots (for example a single repo containing `int4/`, `int8/`, `mf-int4/`, `mf-int8/`) are not supported; pass a split variant repo ID (recommended) or the exact local checkpoint directory for one variant.

## Links

- [:octicons-mark-github-16: Source code](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/dots_tts)
- [:octicons-link-external-16: mlx-community/dots-tts-mlx-int4](https://huggingface.co/mlx-community/dots-tts-mlx-int4)
- [:octicons-link-external-16: mlx-community/dots-tts-mlx-int8](https://huggingface.co/mlx-community/dots-tts-mlx-int8)
- [:octicons-link-external-16: mlx-community/dots-tts-mlx-mf-int4](https://huggingface.co/mlx-community/dots-tts-mlx-mf-int4)
- [:octicons-link-external-16: mlx-community/dots-tts-mlx-mf-int8](https://huggingface.co/mlx-community/dots-tts-mlx-mf-int8)
