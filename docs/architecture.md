# Dia Model Architecture

This document outlines the architecture of the Dia model, as implemented in `dia/model.py`. Dia is designed for text-to-audio generation, potentially conditioned on an audio prompt.

## Overview

The `Dia` class serves as the main interface for interacting with the model. It handles loading the model weights (from local files or Hugging Face Hub), managing the underlying `DiaModel` and the Digital Audio Codec (DAC) model, and orchestrating the audio generation process.

The core generation process involves:
1.  **Text Encoding:** Processing the input text prompt, converting it to tokens, padding, and creating attention masks.
2.  **Encoder Pass:** Feeding the prepared text tokens through the `DiaModel`'s encoder to obtain contextual embeddings.
3.  **Decoder Initialization:** Setting up the Key-Value (KV) caches for self-attention and cross-attention within the decoder. Optionally, pre-filling the decoder state using an audio prompt.
4.  **Autoregressive Decoding:** Generating audio tokens step-by-step using the decoder, conditioned on the encoder output and previously generated tokens. This involves classifier-free guidance (CFG) and sampling strategies (temperature, top-p).
5.  **Audio Synthesis:** Converting the generated audio tokens back into a waveform using the DAC model.

```mermaid
graph TD
    subgraph Inputs
        A[Text Prompt] --> B(Text Processing)
        A_Opt[Audio Prompt Path (Optional)] --> C{Load Audio}
        C --> D[DAC: Audio to Codebook] --> E(Pre-fill Decoder State)
    end

    subgraph CFG Execution
        direction LR
        subgraph Conditional Path
            B -- Tokens/Masks --> F_cond[Encoder]
            F_cond -- Encoder Out --> G_cond[Decoder]
        end
        subgraph Unconditional Path
            Z[Null Input] --> F_uncond[Encoder]
            F_uncond -- Encoder Out --> G_uncond[Decoder]
        end
    end

    subgraph Decoder Logic
        G_cond -- Logits --> H{Combine Logits (CFG)}
        G_uncond -- Logits --> H
        E --> G_cond
        E --> G_uncond
        H --> I(Sampling: Temp, Top-p, CFG Filter)
        I -- Next Token --> J[Generated Audio Tokens]
        J -- Feedback --> G_cond
        J -- Feedback --> G_uncond
        K[KV Cache] <--> G_cond
        K <--> G_uncond
    end

    subgraph Output
        J --> L[DAC: Codebook to Audio]
        L --> M[Audio Waveform]
    end

    classDef dac fill:#f9d,stroke:#333,stroke-width:2px;
    class D,L dac;

    classDef model fill:#ccf,stroke:#333,stroke-width:2px;
    class F_cond,F_uncond,G_cond,G_uncond model
```

## Core Components

### `Dia` Class
-   **Initialization (`__init__`)**: Takes a `DiaConfig` object and a target device. Initializes the `DiaModel`.
-   **Loading (`from_local`, `from_pretrained`)**: Class methods to load model configuration and weights either from local paths or a Hugging Face repository. Handles downloading artifacts if necessary. Ensures the DAC model is loaded (`_load_dac_model`).
-   **DAC Integration (`_load_dac_model`)**: Downloads and loads the pre-trained DAC model required for converting between audio waveforms and discrete codebook representations.
-   **Text Preparation (`_prepare_text_input`)**: Encodes the input text string into byte tokens, handles special tokens (`[S1]`, `[S2]`), pads the sequence to `config.data.text_length`, and creates source token tensors, position tensors, padding masks, and the encoder self-attention mask.
-   **Attention Mask Creation (`_create_attn_mask`)**: Generates attention masks compatible with JAX/TPU conventions. It handles both self-attention (potentially causal) and cross-attention, ensuring that padding tokens attend correctly (or not at all) to other tokens based on their padding status.
-   **Generation (`generate`)**: The main method for generating audio. It orchestrates the entire process described in the Overview section.

### `DiaModel` (Imported from `.layers`)
-   This class (defined elsewhere, likely in `dia/layers.py`) encapsulates the core transformer architecture. It contains:
    -   An **Encoder**: Processes the input text embeddings.
    -   A **Decoder**: Autoregressively generates audio token representations, attending to both the encoder output (cross-attention) and previously generated tokens (self-attention).
-   The `DiaModel` likely uses components like Multi-Head Attention (or Grouped Query Attention, given `gqa_query_heads` in config), Layer Normalization, and Feed-Forward networks.

### DAC Model (Imported from `dac`)
-   A pre-trained model used for audio tokenization and synthesis.
-   **`audio_to_codebook`**: Converts an input audio waveform into a sequence of discrete tokens (codes) across multiple codebooks (channels).
-   **`codebook_to_audio`**: Synthesizes an audio waveform from a sequence of discrete tokens.

## Encoder

-   Takes token IDs (`x_ids`), positional embeddings (`src_positions`), and an attention mask (`attn_mask`) as input.
-   Processes the text input to produce context-rich embeddings (`encoder_out`) of shape `[Batch, SequenceLength, EmbeddingDim]`.
-   The specific architecture (number of layers, heads, etc.) is defined within the `DiaModel` and configured via `DiaConfig`.
-   Uses a non-causal self-attention mechanism, as indicated by `is_causal=False` in `_prepare_text_input`.

## Decoder

-   **Autoregressive Generation**: Generates audio tokens one step at a time.
-   **Inputs**: Takes target token IDs (`tgt_ids_BxTxC`), target positions, encoder output, attention masks, and KV caches.
-   **Attention Mechanisms**:
    -   **Self-Attention**: Attends to previously generated audio tokens in a causal manner. Uses `KVCache` for efficiency during incremental decoding.
    -   **Cross-Attention**: Attends to the `encoder_out` from the encoder. Uses pre-computed KV values (`decoder_cross_attention_cache`) based on the encoder output.
-   **KV Caching**: Employs `KVCache` objects to store past keys and values for both self-attention and cross-attention, significantly speeding up the generation process by avoiding redundant computations. The self-attention cache is updated dynamically at each step, while the cross-attention cache is pre-computed.
-   **Decoding Step (`decode_step`)**: A potentially optimized (e.g., `torch.compile`) function that performs a single decoding step, taking the current token and previous cache states to predict logits for the next token.
-   **Delay Pattern**: Implements a delay pattern (`config.data.delay_pattern`) where tokens for different codebooks (channels) are predicted with specific offsets relative to the first channel. This likely helps model the dependencies between different quantization levels.

## Generation Process Details

### Classifier-Free Guidance (CFG)
-   Uses CFG to improve adherence to the text prompt.
-   Runs the encoder and decoder twice in parallel: once with the actual text conditioning and once with null conditioning (zeroed input tokens).
-   The final logits are calculated as: `cfg_logits = cond_logits + cfg_scale * (cond_logits - uncond_logits)`.
-   The `cfg_scale` parameter controls the strength of the guidance.

### Sampling (`_sample_next_token`)
-   Provides options for sampling the next token from the predicted logits:
    -   **Temperature Scaling**: Divides logits by `temperature` to control randomness (higher temperature -> more random). `temperature=0` implies greedy decoding (argmax).
    -   **Top-p (Nucleus) Sampling**: Selects the next token from the smallest set of tokens whose cumulative probability exceeds `top_p`.
    -   **CFG Filtering**: Optionally applies a filtering step *before* top-p, masking out tokens that are not within the `cfg_filter_top_k` highest probability tokens according to the *conditional* logits. This aims to mitigate potential quality issues sometimes associated with high CFG scales.

### Handling Audio Prompts
-   If an `audio_prompt_path` is provided:
    1.  The audio prompt is loaded, resampled to the target sample rate (44.1kHz), and converted into codebook tokens using the DAC model.
    2.  These prompt tokens are used to pre-fill the beginning of the `generated_BxTxC` tensor.
    3.  A "pre-fill" pass through the decoder is performed using the prompt tokens to populate the self-attention KV cache correctly before starting the autoregressive generation from the end of the prompt.

### EOS Handling and Padding
-   Generation continues until `max_tokens` are generated or an End-of-Sentence (EOS) token (`config.data.audio_eos_value`) is detected in the primary channel (channel 0).
-   After detecting EOS in channel 0, generation continues for a fixed number of `extra_steps_after_eos` steps to allow other channels (based on the `delay_pattern`) to also emit their EOS tokens, followed by padding tokens (`config.data.audio_pad_value`).
-   The final output codes are trimmed to remove the initial BOS token and any trailing padding/EOS tokens generated after the required delay pattern completion.

## Input / Output

-   **Input Text**: A string `text` containing the prompt.
-   **Optional Input Audio**: Path `audio_prompt_path` to a `.wav` or similar audio file.
-   **Output**: A NumPy array representing the generated audio waveform.
