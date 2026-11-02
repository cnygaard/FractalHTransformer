# FractalGPT: Fractal Hierarchical Transformer

A language model architecture that uses multi-resolution fractal attention patterns for efficient autoregressive text generation.

## Overview

FractalGPT organizes transformer layers into a fractal hierarchy with level-specific attention patterns:

- **Level 0 (coarse)**: Strided causal attention -- sparse, long-range context
- **Level 1 (medium)**: Mixed local window + global landmark attention
- **Level 2 (fine)**: Dense local causal attention -- recent context focus

This creates a multi-scale representation where coarse levels capture global structure and fine levels handle local coherence, similar to how fractals exhibit self-similar structure at different scales.

## Architecture

Built on a hierarchical transformer with:

- **Fractal attention patterns**: Each level uses a different causal attention sparsity pattern
- **Hierarchical FFN sizing**: Finer levels get larger feed-forward networks
- **Adaptive computation**: Optional layer skipping at inference time based on activation norms
- **Standard training**: AdamW optimizer with cosine LR schedule

### Model Sizes

| Config | Hidden | Layers | Heads | Params |
|--------|--------|--------|-------|--------|
| Tiny   | 128    | 4      | 2     | ~2M    |
| Small  | 256    | 6      | 4     | ~8M    |
| Medium | 512    | 8      | 8     | ~26M   |

## Training

```bash
# Install dependencies
pip install torch numpy datasets tokenizers tqdm matplotlib jax

# Train on TinyStories (recommended for convergence)
python train_fractal_voronoi_combined.py --small --tinystories

# Train on mixed TinyStories + WikiText
python train_fractal_voronoi_combined.py --medium

# Train on WikiText-103 only
python train_fractal_voronoi_combined.py --medium --wikitext

# Custom settings
python train_fractal_voronoi_combined.py --medium --tinystories --steps 5000 --epochs 20 --lr 3e-4
```

### CLI Options

- `--tiny`, `--small`, `--medium`: Model size presets
- `--tinystories`: Train on TinyStories only (simpler distribution, lower achievable loss)
- `--wikitext`: Train on WikiText-103 only, with the byte-level BPE tokenizer `fractal_gpt_tokenizer_wikitext_8k.json` (rebuild it with `python train_wikitext_tokenizer.py`). Validation also uses WikiText, so losses aren't comparable to TinyStories runs
- `--steps N`: Steps per epoch
- `--epochs N`: Number of epochs
- `--lr RATE`: Learning rate
- `--resume PATH`: Resume from checkpoint

## Core Files

- `fractal_gpt_production.py` -- Model architecture (attention, layers, FractalGPT)
- `llm_fractal_training.py` -- Configuration and fractal layer distribution
- `train_fractal_voronoi_combined.py` -- Training script with data loading and evaluation
- `wikitext_data.py` -- WikiText detokenization and token-stream windowing
- `train_wikitext_tokenizer.py` -- Trains the WikiText-103 tokenizer
- `ellipsoid_core.py` -- Ellipsoid geometry primitives
- `fractal_tree.py` -- Fractal tree decomposition
- `adaptive_moments.py` -- Hierarchical moment tracking

## Results

Small model (8M params, 8 epochs on TinyStories + WikiText):
- Validation loss: **2.82**
- Validation perplexity: **16.8**
- Generates coherent short stories from prompts

## Theoretical Foundations

The fractal attention hierarchy is motivated by:

1. **Multi-resolution analysis**: Different layers attend at different scales, analogous to wavelet decomposition
2. **Fractal self-similarity**: The hierarchical layer structure mirrors fractal geometry where each level has a structurally similar but differently-scaled attention pattern
3. **Ellipsoid decomposition**: Parameter space geometry is modeled via ellipsoid primitives for hierarchical optimization

## License

MIT
