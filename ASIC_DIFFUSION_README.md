# ASIC-DIFFUSION Experiment v1.0

## Project C.H.R.O.M.A. - Cryptographic Hardware Reservoir for Organic Media Art

**Author:** Francisco Angulo de Lafuente  
**GitHub:** https://github.com/Agnuxo1  
**ResearchGate:** https://www.researchgate.net/profile/Francisco-Angulo-Lafuente-3  
**Kaggle:** https://www.kaggle.com/franciscoangulo  
**HuggingFace:** https://huggingface.co/Agnuxo  
**Wikipedia:** https://es.wikipedia.org/wiki/Francisco_Angulo_de_Lafuente

---

## Scientific Integrity Statement

This experiment is designed with **SCIENTIFIC HONESTY** as the primary principle.

### What This Experiment Proves

1. The LV06 ASIC generates deterministic, reproducible hash sequences
2. SHA-256 output exhibits specific statistical properties (bit distribution, autocorrelation, avalanche effect)
3. ASIC-generated data can be transformed into visual textures
4. Energy efficiency comparison with software PRNG

### What This Experiment Does NOT Prove

1. That ASIC-generated latent spaces produce "better" art than Gaussian noise
2. That this replaces standard diffusion model initialization
3. That the visual outputs have artistic merit (subjective)
4. Equivalence to properly trained diffusion models

### Limitations Explicitly Acknowledged

- Visual texture analysis is qualitative, not quantitative
- Diffusion model integration requires pre-trained models (optional)
- "Artistic quality" is not objectively measurable
- The ASIC generates deterministic chaos, not true randomness

---

## Requirements

### Hardware

- **Lucky Miner LV06** (BM1366 ASIC, ~500 GH/s SHA-256)
- USB connection to host computer
- Network connectivity (same LAN)

### Software

```bash
# Required
Python 3.8+

# Recommended
pip install numpy pillow

# Optional (for diffusion model integration)
pip install torch diffusers
```

---

## Experiment Overview

The experiment runs five phases:

### Phase 1: Warmup
Thermal stabilization of the ASIC (3 hashes).

### Phase 2: Software Benchmark
Baseline performance measurement of software PRNG methods.

### Phase 3: Statistical Analysis
Comprehensive randomness tests:
- **Bit Frequency Test**: Verifies ~50% ones/zeros
- **Byte Distribution Test**: Verifies uniform byte values
- **Runs Test**: Verifies proper run length distribution
- **Autocorrelation Test**: Verifies independence between bits
- **Avalanche Test**: Verifies ~50% bit flips on 1-bit input change

### Phase 4: Texture Generation (Project CHROMA)
Visual texture creation:
- **Grayscale**: 512×512 image from 8,192 hashes
- **RGB**: 512×512 color image from 24,576 hashes
- **Analysis**: Entropy, FFT spectrum, local variance

### Phase 5: Latent Space Construction
Build diffusion model-compatible tensors:
- **SD 1.5 format**: (1, 4, 64, 64) = 16,384 values
- **Comparison**: ASIC (uniform) vs Gaussian (normal) distribution

---

## How to Run

### Step 1: Start the Experiment

```bash
python asic_diffusion_experiment_lv06.py
```

### Step 2: Configure Your LV06

Configure your miner to connect to the displayed IP:

```
Pool URL: stratum+tcp://YOUR_IP:3333
Worker: any_name
Password: x
```

### Step 3: Wait for Results

The experiment runs automatically (~15-30 minutes depending on hash collection).

---

## Output Files

Results are saved to `asic_diffusion_results/`:

```
asic_diffusion_results/
├── experiment_summary_YYYYMMDD_HHMMSS.json   # Complete results
├── hash_samples_YYYYMMDD_HHMMSS.csv          # Individual hash data
├── chroma_grayscale_TIMESTAMP.png            # Grayscale texture
├── chroma_rgb_TIMESTAMP.png                  # RGB texture
└── asic_latent_sd15_TIMESTAMP.npy            # SD 1.5 latent tensor
```

---

## Understanding the Results

### Statistical Tests

All tests should **PASS** for valid ASIC output:

| Test | Expected | Meaning |
|------|----------|---------|
| Bit Frequency | ~0.50 | Equal ones and zeros |
| Byte Distribution | χ² < 293 | Uniform byte values |
| Runs Test | \|z\| < 1.96 | Proper run lengths |
| Autocorrelation | < 0.05 | No bit dependencies |
| Avalanche | ~128 bits | Proper diffusion |

### Texture Analysis

```
Entropy: ~7.99 / 8.00 (higher = more random)
FFT peak ratio: < 100 (lower = less pattern)
Appears random: True
```

### Latent Comparison

```
ASIC distribution:  Uniform [-1, 1]  → std ≈ 0.577
Gaussian distribution: Normal N(0,1) → std ≈ 1.0
```

**Note:** Diffusion models expect Gaussian noise. ASIC produces uniform distribution. This may produce different (not necessarily worse) results.

---

## Configuration

Edit `Config` class to modify parameters:

```python
class Config:
    # Network
    HOST = "0.0.0.0"
    PORT = 3333
    
    # Canvas size
    CANVAS_WIDTH = 512
    CANVAS_HEIGHT = 512
    
    # Difficulty (lower = faster hashes)
    DIFFICULTY = 0.0001
```

---

## Using the Latent with Stable Diffusion

The generated `.npy` file can be used with Stable Diffusion:

```python
import numpy as np
import torch
from diffusers import StableDiffusionPipeline

# Load ASIC latent
asic_latent = np.load("asic_latent_sd15_TIMESTAMP.npy")
latent_tensor = torch.from_numpy(asic_latent).to("cuda")

# Load pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
).to("cuda")

# Generate with ASIC latent (experimental)
# Note: May require latent scaling adjustments
```

---

## Energy Efficiency

The LV06 at 3.5W is significantly more efficient than CPU:

| Method | Power | Speed | J/hash |
|--------|-------|-------|--------|
| LV06 ASIC | 3.5W | 500 GH/s | ~0.000000007 |
| CPU (est.) | 50W | ~20K H/s | ~0.0025 |

**Note:** ASIC latency in experiment includes communication overhead. Raw hash computation is orders of magnitude faster.

---

## Scientific Notes

### Why Uniform vs Gaussian Matters

Diffusion models are trained with Gaussian noise N(0,1). Our ASIC produces uniform distribution U(-1,1). The distributions differ:

- **Gaussian**: Bell curve, unbounded tails
- **Uniform**: Flat distribution, bounded range

This doesn't mean ASIC latents are "wrong" - they represent a different exploration of latent space. Future work could:

1. Learn a mapping from uniform to Gaussian
2. Train diffusion models directly with uniform noise
3. Explore hybrid approaches

### Determinism Advantage

Unlike software PRNG, ASIC hashes are:
- **Absolutely reproducible**: Same seed → same hash forever
- **Hardware-verified**: Cryptographic proof of computation
- **Energy-efficient**: Orders of magnitude less power

---

## License

MIT License

---

## Citation

```bibtex
@software{angulo2024asic_diffusion,
  author = {Angulo de Lafuente, Francisco},
  title = {ASIC-DIFFUSION: Hardware-Generated Latent Spaces},
  year = {2024},
  url = {https://github.com/Agnuxo1}
}
```

---

## Contact

- **GitHub:** https://github.com/Agnuxo1
- **ResearchGate:** https://www.researchgate.net/profile/Francisco-Angulo-Lafuente-3
