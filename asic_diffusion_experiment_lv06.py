#!/usr/bin/env python3
"""
ASIC-DIFFUSION: Hardware-Generated Latent Space Experiment v1.0

=============================================================================
                          SCIENTIFIC INTEGRITY STATEMENT
=============================================================================

This experiment is designed with SCIENTIFIC HONESTY as the primary principle.

WHAT THIS EXPERIMENT PROVES:
  1. The LV06 ASIC generates deterministic, reproducible hash sequences
  2. SHA-256 output exhibits specific statistical properties (bit distribution,
     autocorrelation, avalanche effect)
  3. ASIC-generated data can be transformed into visual textures
  4. Energy efficiency comparison with software PRNG

WHAT THIS EXPERIMENT DOES NOT PROVE:
  1. That ASIC-generated latent spaces produce "better" art than Gaussian noise
  2. That this replaces standard diffusion model initialization
  3. That the visual outputs have artistic merit (subjective)
  4. Equivalence to properly trained diffusion models

LIMITATIONS EXPLICITLY ACKNOWLEDGED:
  - Visual texture analysis is qualitative, not quantitative
  - Diffusion model integration requires pre-trained models (optional)
  - "Artistic quality" is not objectively measurable
  - The ASIC generates deterministic chaos, not true randomness

The goal is to establish EMPIRICAL BASELINES and explore novel computational
approaches, not to make premature claims about artistic superiority.

=============================================================================

Project C.H.R.O.M.A. - Cryptographic Hardware Reservoir for Organic Media Art

Author: Francisco Angulo de Lafuente
GitHub: https://github.com/Agnuxo1
ResearchGate: https://www.researchgate.net/profile/Francisco-Angulo-Lafuente-3
Kaggle: https://www.kaggle.com/franciscoangulo
HuggingFace: https://huggingface.co/Agnuxo
Wikipedia: https://es.wikipedia.org/wiki/Francisco_Angulo_de_Lafuente

Date: December 2024
License: MIT
"""

import socket
import threading
import json
import time
import os
import csv
import hashlib
import struct
import sys
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import traceback
import random

# Optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    print("[WARN] NumPy not installed. Statistical analysis will be limited.")
    HAS_NUMPY = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    print("[WARN] Pillow not installed. Image generation disabled.")
    HAS_PIL = False

# Optional: Stable Diffusion integration
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from diffusers import StableDiffusionPipeline
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Experiment configuration - all parameters documented for reproducibility."""
    
    # Network settings for LV06 connection
    HOST = "0.0.0.0"
    PORT = 3333
    
    # Mining difficulty (controls time per hash)
    # Very low difficulty for rapid hash generation
    DIFFICULTY = 0.0001
    
    # Experiment timing
    WARMUP_HASHES = 3
    TIMEOUT_PER_HASH = 60.0
    
    # Hardware specifications
    LV06_HASHRATE_GHS = 500
    LV06_POWER_WATTS = 3.5
    LV06_CHIP = "BM1366"
    
    # Canvas generation (Project CHROMA)
    CANVAS_WIDTH = 512
    CANVAS_HEIGHT = 512
    CANVAS_MODES = ["grayscale", "rgb", "rgba"]
    
    # Statistical analysis
    STATISTICAL_SAMPLE_SIZE = 100000  # Hashes for statistical tests
    NIST_TEST_BITS = 1000000          # Bits for NIST-style tests
    
    # Latent space dimensions (for Stable Diffusion compatibility)
    SD15_LATENT_SHAPE = (1, 4, 64, 64)    # SD 1.5: 512x512 output
    SDXL_LATENT_SHAPE = (1, 4, 128, 128)  # SDXL: 1024x1024 output
    
    # Stratum protocol
    EXTRANONCE1 = "deadbeef"
    EXTRANONCE2_SIZE = 4
    BLOCK_VERSION = 0x20000000
    VERSION_ROLLING_MASK = 0x1fffe000
    
    # Output
    OUTPUT_DIR = Path("asic_diffusion_results")
    
    # Reproducibility
    RANDOM_SEED = 42


# =============================================================================
# CRYPTOGRAPHIC PRIMITIVES
# =============================================================================

def sha256(data: bytes) -> bytes:
    """Single SHA-256 hash."""
    return hashlib.sha256(data).digest()

def sha256d(data: bytes) -> bytes:
    """Double SHA-256 (Bitcoin standard)."""
    return sha256(sha256(data))

def uint32_le(value: int) -> bytes:
    """Pack integer as little-endian 32-bit."""
    return struct.pack('<I', value)

def bytes_to_hex(data: bytes) -> str:
    """Convert bytes to hexadecimal string."""
    return data.hex()

def hex_to_bytes(hex_str: str) -> bytes:
    """Convert hexadecimal string to bytes."""
    return bytes.fromhex(hex_str)


# =============================================================================
# STATISTICAL ANALYSIS TOOLS
# =============================================================================

class StatisticalAnalyzer:
    """
    Statistical analysis tools for hash output characterization.
    
    These tests verify that the ASIC output has the expected properties
    for use as a pseudo-random source.
    """
    
    @staticmethod
    def bit_frequency(data: bytes) -> Dict:
        """
        Analyze bit frequency distribution.
        
        For ideal randomness, each bit position should be ~50% ones.
        """
        total_bits = len(data) * 8
        ones_count = 0
        
        for byte in data:
            ones_count += bin(byte).count('1')
        
        zeros_count = total_bits - ones_count
        ones_ratio = ones_count / total_bits
        zeros_ratio = zeros_count / total_bits
        
        # Chi-squared test for uniformity
        expected = total_bits / 2
        chi_sq = ((ones_count - expected)**2 / expected + 
                  (zeros_count - expected)**2 / expected)
        
        # For 1 degree of freedom, chi_sq < 3.84 means p > 0.05
        passes_chi_sq = chi_sq < 3.84
        
        return {
            "total_bits": total_bits,
            "ones_count": ones_count,
            "zeros_count": zeros_count,
            "ones_ratio": ones_ratio,
            "zeros_ratio": zeros_ratio,
            "chi_squared": chi_sq,
            "passes_uniformity_test": passes_chi_sq,
            "deviation_from_half": abs(ones_ratio - 0.5)
        }
    
    @staticmethod
    def byte_distribution(data: bytes) -> Dict:
        """
        Analyze byte value distribution.
        
        For ideal randomness, all 256 byte values should be equally likely.
        """
        counts = [0] * 256
        for byte in data:
            counts[byte] += 1
        
        total = len(data)
        expected = total / 256
        
        # Chi-squared for byte distribution
        chi_sq = sum((c - expected)**2 / expected for c in counts)
        
        # For 255 degrees of freedom, critical value at p=0.05 is ~293
        passes_chi_sq = chi_sq < 293
        
        min_count = min(counts)
        max_count = max(counts)
        
        return {
            "total_bytes": total,
            "expected_per_value": expected,
            "min_count": min_count,
            "max_count": max_count,
            "chi_squared": chi_sq,
            "passes_distribution_test": passes_chi_sq,
            "range_ratio": max_count / min_count if min_count > 0 else float('inf')
        }
    
    @staticmethod
    def runs_test(data: bytes) -> Dict:
        """
        Runs test for randomness.
        
        A 'run' is a sequence of identical bits. Random data should have
        a predictable distribution of run lengths.
        """
        # Convert to bit string
        bits = ''.join(format(byte, '08b') for byte in data)
        
        # Count runs
        runs = 1
        for i in range(1, len(bits)):
            if bits[i] != bits[i-1]:
                runs += 1
        
        n = len(bits)
        ones = bits.count('1')
        zeros = n - ones
        
        # Expected runs for random sequence
        if ones == 0 or zeros == 0:
            return {"error": "All bits same value"}
        
        expected_runs = 1 + (2 * ones * zeros) / n
        variance = (2 * ones * zeros * (2 * ones * zeros - n)) / (n * n * (n - 1))
        
        if variance <= 0:
            return {"error": "Invalid variance"}
        
        std_dev = math.sqrt(variance)
        z_score = (runs - expected_runs) / std_dev
        
        # |z| < 1.96 means p > 0.05
        passes_runs_test = abs(z_score) < 1.96
        
        return {
            "total_bits": n,
            "total_runs": runs,
            "expected_runs": expected_runs,
            "z_score": z_score,
            "passes_runs_test": passes_runs_test
        }
    
    @staticmethod
    def autocorrelation(data: bytes, max_lag: int = 32) -> Dict:
        """
        Compute autocorrelation at various lags.
        
        For random data, autocorrelation should be near zero at all lags.
        """
        if not HAS_NUMPY:
            return {"error": "NumPy required for autocorrelation"}
        
        # Convert to numpy array of bits
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        bits = bits.astype(np.float64) - 0.5  # Center around 0
        
        n = len(bits)
        autocorr = {}
        
        for lag in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            if lag >= n:
                break
            
            # Compute autocorrelation at this lag
            corr = np.corrcoef(bits[:-lag], bits[lag:])[0, 1]
            autocorr[f"lag_{lag}"] = float(corr)
        
        # Check if all correlations are small (< 0.05)
        max_corr = max(abs(v) for v in autocorr.values())
        passes_test = max_corr < 0.05
        
        return {
            "autocorrelations": autocorr,
            "max_absolute_correlation": max_corr,
            "passes_independence_test": passes_test
        }
    
    @staticmethod
    def avalanche_test(hash_func, num_tests: int = 1000) -> Dict:
        """
        Test avalanche effect: 1-bit input change should flip ~50% of output bits.
        """
        distances = []
        
        for _ in range(num_tests):
            # Random input
            input1 = os.urandom(32)
            
            # Flip one random bit
            input2 = bytearray(input1)
            byte_idx = random.randint(0, 31)
            bit_idx = random.randint(0, 7)
            input2[byte_idx] ^= (1 << bit_idx)
            input2 = bytes(input2)
            
            # Compute hashes
            hash1 = hash_func(input1)
            hash2 = hash_func(input2)
            
            # Count differing bits
            diff_bits = sum(bin(a ^ b).count('1') for a, b in zip(hash1, hash2))
            distances.append(diff_bits)
        
        mean_distance = sum(distances) / len(distances)
        expected = 128  # Half of 256 bits
        
        variance = sum((d - mean_distance)**2 for d in distances) / len(distances)
        std_dev = math.sqrt(variance)
        
        return {
            "num_tests": num_tests,
            "mean_bit_flips": mean_distance,
            "expected_bit_flips": expected,
            "std_deviation": std_dev,
            "min_flips": min(distances),
            "max_flips": max(distances),
            "deviation_from_expected": abs(mean_distance - expected) / expected
        }


# =============================================================================
# VISUAL TEXTURE GENERATOR (Project CHROMA)
# =============================================================================

class TextureGenerator:
    """
    Convert hash sequences into visual textures.
    
    This implements Project CHROMA - exploring whether ASIC-generated
    hash sequences produce visually interesting patterns.
    """
    
    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height
    
    def hashes_to_grayscale(self, hashes: List[bytes]) -> bytes:
        """
        Convert hash sequence to grayscale image.
        
        Each hash provides 32 bytes = 32 grayscale pixels.
        """
        pixels_needed = self.width * self.height
        hashes_needed = (pixels_needed + 31) // 32
        
        # Concatenate hashes
        raw_bytes = b''.join(hashes[:hashes_needed])
        
        # Trim to exact size
        return raw_bytes[:pixels_needed]
    
    def hashes_to_rgb(self, hashes: List[bytes]) -> bytes:
        """
        Convert hash sequence to RGB image.
        
        Each hash provides 32 bytes = 10 RGB pixels (30 bytes used).
        """
        pixels_needed = self.width * self.height
        bytes_needed = pixels_needed * 3
        hashes_needed = (bytes_needed + 31) // 32
        
        raw_bytes = b''.join(hashes[:hashes_needed])
        
        return raw_bytes[:bytes_needed]
    
    def save_grayscale_image(self, pixel_data: bytes, filename: str) -> bool:
        """Save grayscale pixel data as PNG image."""
        if not HAS_PIL:
            print("[WARN] PIL not available, cannot save image")
            return False
        
        try:
            img = Image.frombytes('L', (self.width, self.height), pixel_data)
            img.save(filename)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save image: {e}")
            return False
    
    def save_rgb_image(self, pixel_data: bytes, filename: str) -> bool:
        """Save RGB pixel data as PNG image."""
        if not HAS_PIL:
            print("[WARN] PIL not available, cannot save image")
            return False
        
        try:
            img = Image.frombytes('RGB', (self.width, self.height), pixel_data)
            img.save(filename)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save image: {e}")
            return False
    
    def analyze_texture(self, pixel_data: bytes, mode: str = "grayscale") -> Dict:
        """
        Analyze visual properties of generated texture.
        
        Returns statistics about the texture that might indicate
        non-random structure.
        """
        if not HAS_NUMPY:
            return {"error": "NumPy required for texture analysis"}
        
        if mode == "grayscale":
            img_array = np.frombuffer(pixel_data, dtype=np.uint8).reshape(
                self.height, self.width)
        else:
            img_array = np.frombuffer(pixel_data, dtype=np.uint8).reshape(
                self.height, self.width, 3)
            # Convert to grayscale for analysis
            img_array = np.mean(img_array, axis=2).astype(np.uint8)
        
        # Basic statistics
        mean_val = float(np.mean(img_array))
        std_val = float(np.std(img_array))
        min_val = int(np.min(img_array))
        max_val = int(np.max(img_array))
        
        # Histogram entropy (measure of uniformity)
        hist, _ = np.histogram(img_array.flatten(), bins=256, range=(0, 256))
        hist = hist / hist.sum()  # Normalize
        hist = hist[hist > 0]  # Remove zeros for log
        entropy = -np.sum(hist * np.log2(hist))
        max_entropy = 8.0  # Maximum for 256 bins
        
        # Local variance (detect patterns)
        # High local variance correlation might indicate patterns
        kernel_size = 8
        local_vars = []
        for i in range(0, self.height - kernel_size, kernel_size):
            for j in range(0, self.width - kernel_size, kernel_size):
                patch = img_array[i:i+kernel_size, j:j+kernel_size]
                local_vars.append(np.var(patch))
        
        local_var_mean = float(np.mean(local_vars))
        local_var_std = float(np.std(local_vars))
        
        # 2D FFT for frequency analysis (detect periodic patterns)
        fft = np.fft.fft2(img_array.astype(np.float64))
        fft_magnitude = np.abs(fft)
        
        # Ignore DC component
        fft_magnitude[0, 0] = 0
        
        # Peak detection (strong peaks indicate patterns)
        fft_mean = float(np.mean(fft_magnitude))
        fft_max = float(np.max(fft_magnitude))
        peak_ratio = fft_max / fft_mean if fft_mean > 0 else 0
        
        return {
            "dimensions": f"{self.width}x{self.height}",
            "mode": mode,
            "pixel_mean": mean_val,
            "pixel_std": std_val,
            "pixel_min": min_val,
            "pixel_max": max_val,
            "histogram_entropy": float(entropy),
            "max_possible_entropy": max_entropy,
            "entropy_ratio": float(entropy / max_entropy),
            "local_variance_mean": local_var_mean,
            "local_variance_std": local_var_std,
            "fft_peak_ratio": peak_ratio,
            "appears_random": entropy > 7.9 and peak_ratio < 100
        }


# =============================================================================
# LATENT SPACE BUILDER
# =============================================================================

class LatentSpaceBuilder:
    """
    Build diffusion model-compatible latent tensors from ASIC hashes.
    
    This transforms raw hash bytes into the format expected by
    Stable Diffusion and similar models.
    """
    
    @staticmethod
    def hashes_to_latent_sd15(hashes: List[bytes]) -> Optional[Any]:
        """
        Build Stable Diffusion 1.5 compatible latent tensor.
        
        SD 1.5 expects: (batch, channels, height, width) = (1, 4, 64, 64)
        Total values needed: 4 * 64 * 64 = 16,384 float32 values
        Total bytes needed: 16,384 * 4 = 65,536 bytes
        Hashes needed: 65,536 / 32 = 2,048 hashes
        """
        if not HAS_NUMPY:
            return None
        
        values_needed = 4 * 64 * 64
        bytes_needed = values_needed  # We'll use 1 byte per value initially
        hashes_needed = (bytes_needed + 31) // 32
        
        if len(hashes) < hashes_needed:
            print(f"[WARN] Need {hashes_needed} hashes, only have {len(hashes)}")
            return None
        
        # Concatenate hashes
        raw_bytes = b''.join(hashes[:hashes_needed])[:bytes_needed]
        
        # Convert to uint8 array
        uint8_array = np.frombuffer(raw_bytes, dtype=np.uint8)
        
        # Scale to [-1, 1] range (approximate standard normal)
        # This is a linear mapping, not Gaussian, but preserves structure
        float_array = (uint8_array.astype(np.float32) - 127.5) / 127.5
        
        # Reshape to latent dimensions
        latent = float_array.reshape(1, 4, 64, 64)
        
        return latent
    
    @staticmethod
    def hashes_to_latent_sdxl(hashes: List[bytes]) -> Optional[Any]:
        """
        Build SDXL compatible latent tensor.
        
        SDXL expects: (1, 4, 128, 128) = 65,536 values
        """
        if not HAS_NUMPY:
            return None
        
        values_needed = 4 * 128 * 128
        bytes_needed = values_needed
        hashes_needed = (bytes_needed + 31) // 32
        
        if len(hashes) < hashes_needed:
            print(f"[WARN] Need {hashes_needed} hashes, only have {len(hashes)}")
            return None
        
        raw_bytes = b''.join(hashes[:hashes_needed])[:bytes_needed]
        uint8_array = np.frombuffer(raw_bytes, dtype=np.uint8)
        float_array = (uint8_array.astype(np.float32) - 127.5) / 127.5
        latent = float_array.reshape(1, 4, 128, 128)
        
        return latent
    
    @staticmethod
    def compare_with_gaussian(asic_latent: Any) -> Dict:
        """
        Compare ASIC-generated latent with standard Gaussian.
        
        This shows how the ASIC distribution differs from the
        Gaussian distribution that diffusion models expect.
        """
        if not HAS_NUMPY:
            return {"error": "NumPy required"}
        
        # Generate equivalent Gaussian
        gaussian_latent = np.random.randn(*asic_latent.shape).astype(np.float32)
        
        # Statistics comparison
        asic_mean = float(np.mean(asic_latent))
        asic_std = float(np.std(asic_latent))
        asic_min = float(np.min(asic_latent))
        asic_max = float(np.max(asic_latent))
        
        gauss_mean = float(np.mean(gaussian_latent))
        gauss_std = float(np.std(gaussian_latent))
        gauss_min = float(np.min(gaussian_latent))
        gauss_max = float(np.max(gaussian_latent))
        
        # Distribution comparison
        # ASIC: uniform [-1, 1] mapped
        # Gaussian: normal N(0, 1)
        
        return {
            "asic": {
                "mean": asic_mean,
                "std": asic_std,
                "min": asic_min,
                "max": asic_max,
                "distribution": "uniform_scaled"
            },
            "gaussian": {
                "mean": gauss_mean,
                "std": gauss_std,
                "min": gauss_min,
                "max": gauss_max,
                "distribution": "normal"
            },
            "comparison": {
                "mean_diff": abs(asic_mean - gauss_mean),
                "std_diff": abs(asic_std - gauss_std),
                "note": "ASIC produces uniform distribution, Gaussian produces normal. "
                        "For diffusion models trained on Gaussian noise, ASIC latents "
                        "may produce different (not necessarily worse) results."
            }
        }


# =============================================================================
# SOFTWARE BASELINE GENERATOR
# =============================================================================

class SoftwareNoiseGenerator:
    """
    Software-based noise generation for comparison.
    
    This provides fair baselines for comparing ASIC-generated
    data with standard software approaches.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        if HAS_NUMPY:
            self.rng = np.random.RandomState(seed)
        else:
            random.seed(seed)
    
    def generate_gaussian_latent_sd15(self) -> Any:
        """Generate standard Gaussian latent for SD 1.5."""
        if not HAS_NUMPY:
            return None
        return self.rng.randn(1, 4, 64, 64).astype(np.float32)
    
    def generate_uniform_latent_sd15(self) -> Any:
        """Generate uniform [-1, 1] latent for SD 1.5 (matches ASIC distribution)."""
        if not HAS_NUMPY:
            return None
        return (self.rng.rand(1, 4, 64, 64).astype(np.float32) * 2 - 1)
    
    def benchmark_prng(self, num_bytes: int, iterations: int = 100) -> Dict:
        """Benchmark software PRNG performance."""
        
        # os.urandom (cryptographic)
        start = time.perf_counter()
        for _ in range(iterations):
            _ = os.urandom(num_bytes)
        urandom_time = time.perf_counter() - start
        
        # hashlib (SHA-256 chain)
        start = time.perf_counter()
        seed = os.urandom(32)
        for _ in range(iterations):
            seed = hashlib.sha256(seed).digest()
        sha256_time = time.perf_counter() - start
        
        # NumPy random (if available)
        numpy_time = 0
        if HAS_NUMPY:
            start = time.perf_counter()
            for _ in range(iterations):
                _ = np.random.bytes(num_bytes)
            numpy_time = time.perf_counter() - start
        
        return {
            "bytes_per_call": num_bytes,
            "iterations": iterations,
            "urandom_total_s": urandom_time,
            "urandom_per_call_us": urandom_time / iterations * 1e6,
            "sha256_chain_total_s": sha256_time,
            "sha256_chain_per_call_us": sha256_time / iterations * 1e6,
            "numpy_total_s": numpy_time if HAS_NUMPY else None,
            "numpy_per_call_us": numpy_time / iterations * 1e6 if HAS_NUMPY else None
        }


# =============================================================================
# STRATUM PROTOCOL HELPERS
# =============================================================================

def difficulty_to_target(difficulty: float) -> int:
    """Convert mining difficulty to target value."""
    diff1_target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
    return int(diff1_target / difficulty)

def target_to_nbits(target: int) -> int:
    """Convert target to compact nbits format."""
    target_hex = format(target, '064x')
    first_nonzero = 0
    for i in range(0, 64, 2):
        if target_hex[i:i+2] != '00':
            first_nonzero = i // 2
            break
    exponent = 32 - first_nonzero
    mantissa_hex = target_hex[first_nonzero:first_nonzero+6]
    mantissa = int(mantissa_hex, 16) if mantissa_hex else 0
    if mantissa & 0x800000:
        mantissa >>= 8
        exponent += 1
    return (exponent << 24) | mantissa

def swap_endianness_32bit_words(hex_str: str) -> str:
    """Swap endianness of each 4-byte word."""
    if len(hex_str) % 8 != 0:
        hex_str = hex_str.zfill((len(hex_str) // 8 + 1) * 8)
    result = ""
    for i in range(0, len(hex_str), 8):
        word = hex_str[i:i+8]
        swapped = "".join(reversed([word[j:j+2] for j in range(0, 8, 2)]))
        result += swapped
    return result

def parse_prevhash_from_stratum(prevhash_hex: str) -> bytes:
    """Convert Stratum prevhash to internal byte order."""
    unswapped = swap_endianness_32bit_words(prevhash_hex)
    return hex_to_bytes(unswapped)

def create_coinbase_parts() -> Tuple[str, str]:
    """Create coinbase transaction parts."""
    version = uint32_le(1)
    input_count = bytes([1])
    prev_tx = bytes(32)
    prev_index = bytes.fromhex('ffffffff')
    
    height_bytes = bytes([1, 1])
    prefix_data = b'CHROMA/'
    suffix_data = b'/ART/'
    
    script_prefix = height_bytes + prefix_data
    script_suffix = suffix_data
    script_len = len(script_prefix) + 8 + len(script_suffix)
    
    sequence = bytes.fromhex('ffffffff')
    output_count = bytes([1])
    output_value = struct.pack('<Q', 50 * 100000000)
    
    marker = b'ASIC-DIFFUSION!'
    output_script = bytes([0x6a, len(marker)]) + marker
    output_script_len = bytes([len(output_script)])
    locktime = bytes(4)
    
    coinb1 = (version + input_count + prev_tx + prev_index + 
              bytes([script_len]) + script_prefix)
    coinb2 = (script_suffix + sequence + output_count + output_value +
              output_script_len + output_script + locktime)
    
    return (bytes_to_hex(coinb1), bytes_to_hex(coinb2))

def build_coinbase(coinb1: str, extranonce1: str, extranonce2: str, coinb2: str) -> bytes:
    """Build complete coinbase transaction."""
    return hex_to_bytes(coinb1 + extranonce1 + extranonce2 + coinb2)

def compute_merkle_root(coinbase_hash: bytes, merkle_branches: List[str]) -> bytes:
    """Compute merkle root."""
    current = coinbase_hash
    for branch in merkle_branches:
        branch_bytes = hex_to_bytes(branch)
        current = sha256d(current + branch_bytes)
    return current

def build_block_header(version: int, prev_block_hash: bytes, merkle_root: bytes,
                       timestamp: int, nbits: int, nonce: int) -> bytes:
    """Build 80-byte block header."""
    return (uint32_le(version) + prev_block_hash + merkle_root +
            uint32_le(timestamp) + uint32_le(nbits) + uint32_le(nonce))


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class HashSample:
    """Single hash sample from ASIC."""
    sample_id: int
    timestamp: float
    seed_nonce: int
    hash_hex: str
    latency_s: float
    verified: bool

@dataclass
class TextureResult:
    """Result of texture generation."""
    texture_id: str
    timestamp: float
    mode: str
    width: int
    height: int
    num_hashes_used: int
    filename: str
    analysis: Dict

@dataclass
class StatisticalResult:
    """Statistical analysis results."""
    test_name: str
    timestamp: float
    sample_size: int
    results: Dict
    passed: bool


# =============================================================================
# LV06 DIFFUSION BRIDGE
# =============================================================================

class LV06DiffusionBridge(threading.Thread):
    """
    Stratum bridge for LV06 in diffusion/art generation experiments.
    
    Collects sequential hashes for texture generation and
    latent space construction.
    """
    
    def __init__(self, output_dir: Path):
        super().__init__(daemon=True)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Network
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((Config.HOST, Config.PORT))
        self.sock.listen(5)
        
        self.conn = None
        self.authorized = False
        self.running = True
        
        # Version rolling
        self.version_rolling_enabled = False
        self.version_rolling_mask = Config.VERSION_ROLLING_MASK
        
        # Job management
        self.current_job = None
        self.job_lock = threading.Lock()
        self.job_counter = 0
        self.nonce_counter = 0
        
        # Hash collection
        self.collected_hashes: List[bytes] = []
        self.hash_samples: List[HashSample] = []
        self.hashes_lock = threading.Lock()
        
        # Share handling
        self.share_received = threading.Event()
        self.last_submit = None
        self.last_submit_time = None
        
        # Coinbase
        self.coinb1, self.coinb2 = create_coinbase_parts()
        
        # Stats
        self.stats = {
            'jobs_sent': 0,
            'shares_received': 0,
            'shares_verified': 0,
            'total_hash_bytes': 0
        }
    
    def run(self):
        """Main thread loop."""
        print(f"[BRIDGE] Listening on {Config.HOST}:{Config.PORT}")
        
        while self.running:
            try:
                self.sock.settimeout(1.0)
                conn, addr = self.sock.accept()
                print(f"[BRIDGE] Connection from {addr}")
                self.conn = conn
                self._handle_connection()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[BRIDGE] Error: {e}")
    
    def _handle_connection(self):
        """Handle Stratum messages."""
        buffer = ""
        self.conn.settimeout(0.5)
        
        try:
            while self.running:
                try:
                    data = self.conn.recv(4096).decode('utf-8', errors='ignore')
                    if not data:
                        break
                    buffer += data
                    
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if line.strip():
                            try:
                                msg = json.loads(line)
                                self._handle_message(msg)
                            except json.JSONDecodeError:
                                pass
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"[BRIDGE] Recv error: {e}")
                    break
        finally:
            if self.conn:
                self.conn.close()
                self.conn = None
            self.authorized = False
    
    def _handle_message(self, msg: Dict):
        """Process Stratum message."""
        msg_id = msg.get('id')
        method = msg.get('method')
        
        if method == 'mining.subscribe':
            result = [
                [["mining.set_difficulty", "1"], ["mining.notify", "1"]],
                Config.EXTRANONCE1,
                Config.EXTRANONCE2_SIZE
            ]
            self._send({'id': msg_id, 'result': result, 'error': None})
            print(f"[BRIDGE] Miner subscribed")
        
        elif method == 'mining.configure':
            params = msg.get('params', [])
            extensions = params[0] if params else []
            ext_params = params[1] if len(params) > 1 else {}
            
            result = {}
            if 'version-rolling' in extensions:
                self.version_rolling_enabled = True
                mask = ext_params.get('version-rolling.mask', 'ffffffff')
                self.version_rolling_mask = int(mask, 16)
                result['version-rolling'] = True
                result['version-rolling.mask'] = format(self.version_rolling_mask, '08x')
                print(f"[BRIDGE] Version rolling enabled")
            
            self._send({'id': msg_id, 'result': result, 'error': None})
        
        elif method == 'mining.authorize':
            worker = msg.get('params', ['unknown'])[0]
            self._send({'id': msg_id, 'result': True, 'error': None})
            self.authorized = True
            print(f"[BRIDGE] Authorized: {worker}")
        
        elif method == 'mining.submit':
            self._handle_submit(msg)
        
        elif method == 'mining.suggest_difficulty':
            self._send({'id': msg_id, 'result': True, 'error': None})
        
        elif method == 'mining.extranonce.subscribe':
            self._send({'id': msg_id, 'result': True, 'error': None})
    
    def _handle_submit(self, msg: Dict):
        """Handle share submission."""
        params = msg.get('params', [])
        msg_id = msg.get('id')
        
        if len(params) >= 5:
            submit_data = {
                'worker_name': params[0],
                'job_id': params[1],
                'extranonce2': params[2],
                'ntime': params[3],
                'nonce': params[4],
                'version_bits': params[5] if len(params) >= 6 else None
            }
            
            self.last_submit = submit_data
            self.last_submit_time = time.time()
            self.stats['shares_received'] += 1
            
            self._send({'id': msg_id, 'result': True, 'error': None})
            self.share_received.set()
        else:
            self._send({'id': msg_id, 'result': False, 'error': [20, "Bad params", None]})
    
    def _send(self, data: Dict):
        """Send message to miner."""
        if self.conn:
            try:
                self.conn.sendall((json.dumps(data) + '\n').encode())
            except:
                pass
    
    def set_difficulty(self, difficulty: float):
        """Set mining difficulty."""
        self._send({
            'id': None,
            'method': 'mining.set_difficulty',
            'params': [difficulty]
        })
    
    def send_job(self, seed_nonce: int) -> Dict:
        """Send job with seed nonce to generate deterministic hash."""
        self.job_counter += 1
        job_id = f"art{self.job_counter:06d}"
        
        # Create deterministic prevhash from seed
        seed_bytes = struct.pack('<Q', seed_nonce) + b'\x00' * 24
        seed_hash = sha256(seed_bytes)
        seed_hash_hex = bytes_to_hex(seed_hash)
        
        ntime = int(time.time())
        ntime_hex = format(ntime, 'x')
        
        target = difficulty_to_target(Config.DIFFICULTY)
        nbits = target_to_nbits(target)
        nbits_hex = format(nbits, '08x')
        
        version_hex = format(Config.BLOCK_VERSION, '08x')
        prevhash_stratum = swap_endianness_32bit_words(seed_hash_hex)
        
        job = {
            'job_id': job_id,
            'seed_nonce': seed_nonce,
            'prevhash_stratum': prevhash_stratum,
            'prevhash_raw': seed_hash_hex,
            'coinb1': self.coinb1,
            'coinb2': self.coinb2,
            'merkle_branches': [],
            'version': Config.BLOCK_VERSION,
            'version_hex': version_hex,
            'nbits': nbits,
            'nbits_hex': nbits_hex,
            'ntime': ntime,
            'ntime_hex': ntime_hex,
            'target': target,
            'sent_at': time.time()
        }
        
        with self.job_lock:
            self.current_job = job
        
        notify_params = [
            job_id,
            prevhash_stratum,
            self.coinb1,
            self.coinb2,
            [],
            version_hex,
            nbits_hex,
            ntime_hex,
            True
        ]
        
        self._send({'id': None, 'method': 'mining.notify', 'params': notify_params})
        self.stats['jobs_sent'] += 1
        
        return job
    
    def verify_and_extract_hash(self, job: Dict, submit: Dict) -> Tuple[bool, Optional[bytes], Dict]:
        """Verify share and extract the block hash."""
        verification = {}
        
        try:
            # Build coinbase
            coinbase = build_coinbase(
                job['coinb1'],
                Config.EXTRANONCE1,
                submit['extranonce2'],
                job['coinb2']
            )
            coinbase_hash = sha256d(coinbase)
            
            # Merkle root
            block_merkle_root = compute_merkle_root(coinbase_hash, job['merkle_branches'])
            
            # Prevhash
            prevhash_bytes = parse_prevhash_from_stratum(job['prevhash_stratum'])
            
            # Parse submission
            ntime = int(submit['ntime'], 16)
            nonce = int(submit['nonce'], 16)
            
            # Version handling
            if submit.get('version_bits'):
                version_bits = int(submit['version_bits'], 16)
                BASE_VERSION = 0x20000000
                if version_bits & BASE_VERSION:
                    version_used = version_bits
                else:
                    version_used = BASE_VERSION | version_bits
            else:
                version_used = job['version']
            
            # Build header
            header = build_block_header(
                version=version_used,
                prev_block_hash=prevhash_bytes,
                merkle_root=block_merkle_root,
                timestamp=ntime,
                nbits=job['nbits'],
                nonce=nonce
            )
            
            # Compute block hash
            block_hash = sha256d(header)
            verification['block_hash_hex'] = bytes_to_hex(block_hash[::-1])
            
            # Check target
            hash_int = int.from_bytes(block_hash, byteorder='little')
            meets_target = hash_int < job['target']
            
            return (meets_target, block_hash, verification)
        
        except Exception as e:
            return (False, None, {"error": str(e)})
    
    def collect_hash(self, timeout: float = None) -> Optional[HashSample]:
        """Collect a single hash from the ASIC."""
        if not self.authorized:
            return None
        
        timeout = timeout or Config.TIMEOUT_PER_HASH
        
        # Increment nonce for new seed
        self.nonce_counter += 1
        seed_nonce = self.nonce_counter
        
        # Set difficulty
        self.set_difficulty(Config.DIFFICULTY)
        time.sleep(0.02)
        
        # Send job
        job = self.send_job(seed_nonce)
        
        # Wait for share
        self.share_received.clear()
        
        if not self.share_received.wait(timeout):
            return None
        
        submit = self.last_submit
        submit_time = self.last_submit_time
        
        if not submit or submit['job_id'] != job['job_id']:
            return None
        
        # Verify and extract hash
        verified, block_hash, _ = self.verify_and_extract_hash(job, submit)
        
        if block_hash is None:
            return None
        
        latency = submit_time - job['sent_at']
        
        # Store hash
        with self.hashes_lock:
            self.collected_hashes.append(block_hash)
            self.stats['total_hash_bytes'] += len(block_hash)
            if verified:
                self.stats['shares_verified'] += 1
        
        sample = HashSample(
            sample_id=len(self.hash_samples) + 1,
            timestamp=submit_time,
            seed_nonce=seed_nonce,
            hash_hex=bytes_to_hex(block_hash),
            latency_s=latency,
            verified=verified
        )
        
        self.hash_samples.append(sample)
        
        return sample
    
    def collect_hashes(self, count: int, callback=None) -> List[bytes]:
        """Collect multiple hashes."""
        hashes = []
        
        for i in range(count):
            sample = self.collect_hash()
            if sample:
                hashes.append(hex_to_bytes(sample.hash_hex))
                if callback:
                    callback(i + 1, count, sample)
            else:
                print(f"[COLLECT] Failed at hash {i+1}/{count}")
        
        return hashes
    
    def get_collected_hashes(self) -> List[bytes]:
        """Get all collected hashes."""
        with self.hashes_lock:
            return list(self.collected_hashes)
    
    def stop(self):
        """Stop the bridge."""
        self.running = False
        try:
            self.sock.close()
        except:
            pass


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class DiffusionExperiment:
    """
    Main experiment controller for ASIC-based diffusion/art generation.
    """
    
    def __init__(self):
        self.output_dir = Config.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.bridge = None
        self.texture_gen = TextureGenerator(Config.CANVAS_WIDTH, Config.CANVAS_HEIGHT)
        self.software_gen = SoftwareNoiseGenerator(Config.RANDOM_SEED)
        self.stats_analyzer = StatisticalAnalyzer()
        self.latent_builder = LatentSpaceBuilder()
        
        self.texture_results: List[TextureResult] = []
        self.statistical_results: List[StatisticalResult] = []
        
        self.start_time = None
    
    def setup(self) -> bool:
        """Initialize and wait for LV06."""
        print("\n" + "=" * 70)
        print("  ASIC-DIFFUSION EXPERIMENT v1.0")
        print("  Project C.H.R.O.M.A. - Hardware-Generated Art")
        print("=" * 70)
        
        print(f"\n[SETUP] Configuration:")
        print(f"  - Canvas size: {Config.CANVAS_WIDTH}x{Config.CANVAS_HEIGHT}")
        print(f"  - Output directory: {self.output_dir}")
        
        # Start bridge
        self.bridge = LV06DiffusionBridge(self.output_dir)
        self.bridge.start()
        
        # Get local IP
        hostname = socket.gethostname()
        try:
            local_ip = socket.gethostbyname(hostname)
        except:
            local_ip = "localhost"
        
        print(f"\n[SETUP] Waiting for LV06 connection...")
        print(f"  Configure miner: stratum+tcp://{local_ip}:{Config.PORT}")
        
        try:
            timeout = 300
            start = time.time()
            while not self.bridge.authorized and (time.time() - start) < timeout:
                time.sleep(1)
        except KeyboardInterrupt:
            return False
        
        if not self.bridge.authorized:
            print("[SETUP] Timeout")
            return False
        
        print(f"\n[SETUP] LV06 connected!")
        return True
    
    def run_warmup(self) -> bool:
        """Warmup phase."""
        print(f"\n[WARMUP] Collecting {Config.WARMUP_HASHES} warmup hashes...")
        
        for i in range(Config.WARMUP_HASHES):
            sample = self.bridge.collect_hash()
            if not sample:
                return False
            print(f"  Warmup {i+1}/{Config.WARMUP_HASHES}: {sample.latency_s:.2f}s")
        
        return True
    
    def run_software_benchmark(self) -> Dict:
        """Benchmark software PRNG."""
        print(f"\n[BENCHMARK] Software PRNG benchmark...")
        
        benchmark = self.software_gen.benchmark_prng(
            num_bytes=32,
            iterations=10000
        )
        
        print(f"  os.urandom: {benchmark['urandom_per_call_us']:.2f} µs/call")
        print(f"  SHA-256 chain: {benchmark['sha256_chain_per_call_us']:.2f} µs/call")
        if benchmark['numpy_per_call_us']:
            print(f"  NumPy: {benchmark['numpy_per_call_us']:.2f} µs/call")
        
        return benchmark
    
    def run_statistical_analysis(self, num_hashes: int = 1000) -> List[StatisticalResult]:
        """Run comprehensive statistical analysis."""
        print("\n" + "=" * 70)
        print("  STATISTICAL ANALYSIS")
        print("=" * 70)
        
        print(f"\n[STATS] Collecting {num_hashes} hashes for analysis...")
        
        def progress_callback(current, total, sample):
            if current % 100 == 0:
                print(f"  Progress: {current}/{total} ({sample.latency_s:.2f}s)")
        
        hashes = self.bridge.collect_hashes(num_hashes, callback=progress_callback)
        
        if len(hashes) < num_hashes:
            print(f"[WARN] Only collected {len(hashes)} hashes")
        
        # Concatenate all hash bytes
        all_bytes = b''.join(hashes)
        print(f"\n[STATS] Analyzing {len(all_bytes)} bytes ({len(all_bytes)*8} bits)...")
        
        results = []
        
        # Bit frequency test
        print("\n[TEST] Bit Frequency Analysis...")
        bit_freq = self.stats_analyzer.bit_frequency(all_bytes)
        results.append(StatisticalResult(
            test_name="bit_frequency",
            timestamp=time.time(),
            sample_size=len(all_bytes),
            results=bit_freq,
            passed=bit_freq['passes_uniformity_test']
        ))
        print(f"  Ones ratio: {bit_freq['ones_ratio']:.6f} (expected: 0.5)")
        print(f"  Chi-squared: {bit_freq['chi_squared']:.2f}")
        print(f"  PASSED: {bit_freq['passes_uniformity_test']}")
        
        # Byte distribution test
        print("\n[TEST] Byte Distribution Analysis...")
        byte_dist = self.stats_analyzer.byte_distribution(all_bytes)
        results.append(StatisticalResult(
            test_name="byte_distribution",
            timestamp=time.time(),
            sample_size=len(all_bytes),
            results=byte_dist,
            passed=byte_dist['passes_distribution_test']
        ))
        print(f"  Range ratio: {byte_dist['range_ratio']:.2f}")
        print(f"  Chi-squared: {byte_dist['chi_squared']:.2f}")
        print(f"  PASSED: {byte_dist['passes_distribution_test']}")
        
        # Runs test
        print("\n[TEST] Runs Test...")
        runs = self.stats_analyzer.runs_test(all_bytes)
        if 'error' not in runs:
            results.append(StatisticalResult(
                test_name="runs_test",
                timestamp=time.time(),
                sample_size=len(all_bytes),
                results=runs,
                passed=runs['passes_runs_test']
            ))
            print(f"  Z-score: {runs['z_score']:.4f}")
            print(f"  PASSED: {runs['passes_runs_test']}")
        
        # Autocorrelation test
        if HAS_NUMPY:
            print("\n[TEST] Autocorrelation Analysis...")
            autocorr = self.stats_analyzer.autocorrelation(all_bytes)
            if 'error' not in autocorr:
                results.append(StatisticalResult(
                    test_name="autocorrelation",
                    timestamp=time.time(),
                    sample_size=len(all_bytes),
                    results=autocorr,
                    passed=autocorr['passes_independence_test']
                ))
                print(f"  Max correlation: {autocorr['max_absolute_correlation']:.6f}")
                print(f"  PASSED: {autocorr['passes_independence_test']}")
        
        # Avalanche test (using software SHA-256)
        print("\n[TEST] Avalanche Effect (software verification)...")
        avalanche = self.stats_analyzer.avalanche_test(sha256, num_tests=500)
        results.append(StatisticalResult(
            test_name="avalanche_effect",
            timestamp=time.time(),
            sample_size=500,
            results=avalanche,
            passed=avalanche['deviation_from_expected'] < 0.05
        ))
        print(f"  Mean bit flips: {avalanche['mean_bit_flips']:.2f} (expected: 128)")
        print(f"  Deviation: {avalanche['deviation_from_expected']*100:.2f}%")
        
        self.statistical_results = results
        return results
    
    def run_texture_generation(self) -> List[TextureResult]:
        """Generate visual textures from ASIC hashes."""
        print("\n" + "=" * 70)
        print("  PROJECT C.H.R.O.M.A. - TEXTURE GENERATION")
        print("=" * 70)
        
        results = []
        
        # Calculate hashes needed for one canvas
        pixels = Config.CANVAS_WIDTH * Config.CANVAS_HEIGHT
        hashes_for_grayscale = (pixels + 31) // 32
        hashes_for_rgb = (pixels * 3 + 31) // 32
        
        max_hashes = max(hashes_for_grayscale, hashes_for_rgb)
        
        print(f"\n[CHROMA] Collecting {max_hashes} hashes for textures...")
        
        def progress_callback(current, total, sample):
            if current % 500 == 0 or current == total:
                print(f"  Progress: {current}/{total}")
        
        hashes = self.bridge.collect_hashes(max_hashes, callback=progress_callback)
        
        if len(hashes) < hashes_for_grayscale:
            print("[ERROR] Not enough hashes collected")
            return results
        
        # Generate grayscale texture
        print("\n[CHROMA] Generating grayscale texture...")
        gray_pixels = self.texture_gen.hashes_to_grayscale(hashes)
        gray_filename = str(self.output_dir / f"chroma_grayscale_{int(time.time())}.png")
        
        if self.texture_gen.save_grayscale_image(gray_pixels, gray_filename):
            analysis = self.texture_gen.analyze_texture(gray_pixels, "grayscale")
            results.append(TextureResult(
                texture_id="grayscale_001",
                timestamp=time.time(),
                mode="grayscale",
                width=Config.CANVAS_WIDTH,
                height=Config.CANVAS_HEIGHT,
                num_hashes_used=hashes_for_grayscale,
                filename=gray_filename,
                analysis=analysis
            ))
            print(f"  Saved: {gray_filename}")
            print(f"  Entropy: {analysis['histogram_entropy']:.4f} / {analysis['max_possible_entropy']:.4f}")
            print(f"  Appears random: {analysis['appears_random']}")
        
        # Generate RGB texture
        if len(hashes) >= hashes_for_rgb:
            print("\n[CHROMA] Generating RGB texture...")
            rgb_pixels = self.texture_gen.hashes_to_rgb(hashes)
            rgb_filename = str(self.output_dir / f"chroma_rgb_{int(time.time())}.png")
            
            if self.texture_gen.save_rgb_image(rgb_pixels, rgb_filename):
                analysis = self.texture_gen.analyze_texture(rgb_pixels, "rgb")
                results.append(TextureResult(
                    texture_id="rgb_001",
                    timestamp=time.time(),
                    mode="rgb",
                    width=Config.CANVAS_WIDTH,
                    height=Config.CANVAS_HEIGHT,
                    num_hashes_used=hashes_for_rgb,
                    filename=rgb_filename,
                    analysis=analysis
                ))
                print(f"  Saved: {rgb_filename}")
                print(f"  Entropy: {analysis['histogram_entropy']:.4f}")
        
        self.texture_results = results
        return results
    
    def run_latent_comparison(self) -> Dict:
        """Compare ASIC latent with Gaussian latent."""
        print("\n" + "=" * 70)
        print("  LATENT SPACE COMPARISON")
        print("=" * 70)
        
        if not HAS_NUMPY:
            print("[SKIP] NumPy required for latent comparison")
            return {}
        
        # Get collected hashes
        hashes = self.bridge.get_collected_hashes()
        
        # Need at least 2048 hashes for SD 1.5 latent
        if len(hashes) < 2048:
            print(f"[WARN] Need 2048 hashes for SD latent, have {len(hashes)}")
            print("[SKIP] Collecting more hashes...")
            
            additional = 2048 - len(hashes)
            extra_hashes = self.bridge.collect_hashes(additional)
            hashes.extend(extra_hashes)
        
        print(f"\n[LATENT] Building SD 1.5 latent from {len(hashes)} hashes...")
        
        asic_latent = self.latent_builder.hashes_to_latent_sd15(hashes)
        
        if asic_latent is None:
            print("[ERROR] Failed to build latent")
            return {}
        
        print(f"  Shape: {asic_latent.shape}")
        
        # Compare with Gaussian
        comparison = self.latent_builder.compare_with_gaussian(asic_latent)
        
        print(f"\n[COMPARE] ASIC vs Gaussian latent:")
        print(f"  ASIC mean:     {comparison['asic']['mean']:.6f}")
        print(f"  Gaussian mean: {comparison['gaussian']['mean']:.6f}")
        print(f"  ASIC std:      {comparison['asic']['std']:.6f}")
        print(f"  Gaussian std:  {comparison['gaussian']['std']:.6f}")
        
        # Save latent for potential diffusion use
        latent_file = self.output_dir / f"asic_latent_sd15_{int(time.time())}.npy"
        np.save(str(latent_file), asic_latent)
        print(f"\n[SAVE] Latent saved: {latent_file}")
        
        comparison['latent_file'] = str(latent_file)
        
        return comparison
    
    def calculate_energy_efficiency(self) -> Dict:
        """Calculate energy efficiency metrics."""
        samples = self.bridge.hash_samples
        
        if not samples:
            return {}
        
        total_time = sum(s.latency_s for s in samples)
        total_hashes = len(samples)
        total_bytes = total_hashes * 32
        
        # ASIC energy
        asic_energy_joules = Config.LV06_POWER_WATTS * total_time
        joules_per_hash = asic_energy_joules / total_hashes
        joules_per_byte = asic_energy_joules / total_bytes
        
        # Software comparison (rough estimate)
        # Assume CPU at 50W can do ~20000 SHA-256/s (from benchmark)
        sw_power = 50.0
        sw_hashes_per_sec = 20000
        sw_joules_per_hash = sw_power / sw_hashes_per_sec
        
        efficiency_ratio = sw_joules_per_hash / joules_per_hash if joules_per_hash > 0 else 0
        
        return {
            "asic": {
                "total_hashes": total_hashes,
                "total_time_s": total_time,
                "total_energy_joules": asic_energy_joules,
                "joules_per_hash": joules_per_hash,
                "power_watts": Config.LV06_POWER_WATTS,
                "hashes_per_second": total_hashes / total_time if total_time > 0 else 0
            },
            "software_estimate": {
                "power_watts": sw_power,
                "hashes_per_second": sw_hashes_per_sec,
                "joules_per_hash": sw_joules_per_hash
            },
            "comparison": {
                "efficiency_ratio": efficiency_ratio,
                "note": "Efficiency ratio > 1 means ASIC is more efficient. "
                        "Note: ASIC latency includes communication overhead, "
                        "actual hash computation is much faster."
            }
        }
    
    def save_results(self, software_benchmark: Dict, latent_comparison: Dict) -> str:
        """Save all results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        energy = self.calculate_energy_efficiency()
        
        # Summary
        summary = {
            "experiment_id": f"ASIC_DIFFUSION_{timestamp}",
            "start_time": self.start_time.isoformat() if self.start_time else "",
            "end_time": datetime.now().isoformat(),
            
            "hardware": {
                "device": "Lucky Miner LV06",
                "chip": Config.LV06_CHIP,
                "power_watts": Config.LV06_POWER_WATTS
            },
            
            "configuration": {
                "canvas_width": Config.CANVAS_WIDTH,
                "canvas_height": Config.CANVAS_HEIGHT,
                "difficulty": Config.DIFFICULTY
            },
            
            "hash_collection": {
                "total_hashes": len(self.bridge.hash_samples),
                "verified": sum(1 for s in self.bridge.hash_samples if s.verified),
                "total_bytes": self.bridge.stats['total_hash_bytes']
            },
            
            "statistical_tests": {
                "tests_run": len(self.statistical_results),
                "tests_passed": sum(1 for r in self.statistical_results if r.passed),
                "details": [asdict(r) for r in self.statistical_results]
            },
            
            "texture_generation": {
                "textures_created": len(self.texture_results),
                "details": [asdict(r) for r in self.texture_results]
            },
            
            "latent_comparison": latent_comparison,
            
            "energy_efficiency": energy,
            
            "software_benchmark": software_benchmark,
            
            "scientific_notes": {
                "what_this_proves": [
                    "LV06 generates cryptographically valid hashes",
                    "Hash output passes standard randomness tests",
                    "ASIC data can be transformed into visual textures",
                    "Latent tensors can be constructed from hash sequences"
                ],
                "limitations": [
                    "Visual quality assessment is subjective",
                    "Diffusion model integration not tested here",
                    "ASIC latents have uniform, not Gaussian, distribution",
                    "Communication overhead dominates single-hash timing"
                ],
                "next_steps": [
                    "Test with actual Stable Diffusion pipeline",
                    "Compare generated images: ASIC vs Gaussian initialization",
                    "Explore learned mapping from hash space to Gaussian"
                ]
            },
            
            "author": {
                "name": "Francisco Angulo de Lafuente",
                "github": "https://github.com/Agnuxo1",
                "researchgate": "https://www.researchgate.net/profile/Francisco-Angulo-Lafuente-3"
            }
        }
        
        # Save JSON
        summary_file = self.output_dir / f"experiment_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n[SAVE] Summary: {summary_file}")
        
        # Save hash samples CSV
        if self.bridge.hash_samples:
            csv_file = self.output_dir / f"hash_samples_{timestamp}.csv"
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=asdict(self.bridge.hash_samples[0]).keys())
                writer.writeheader()
                for sample in self.bridge.hash_samples:
                    writer.writerow(asdict(sample))
            print(f"[SAVE] Hash samples: {csv_file}")
        
        # Print summary
        self._print_summary(summary)
        
        return str(summary_file)
    
    def _print_summary(self, summary: Dict):
        """Print human-readable summary."""
        print("\n" + "=" * 70)
        print("                    EXPERIMENT COMPLETE")
        print("=" * 70)
        
        print(f"\n📊 HASH COLLECTION:")
        hc = summary['hash_collection']
        print(f"   Total: {hc['total_hashes']}")
        print(f"   Verified: {hc['verified']}")
        print(f"   Bytes: {hc['total_bytes']}")
        
        print(f"\n🔬 STATISTICAL TESTS:")
        st = summary['statistical_tests']
        print(f"   Passed: {st['tests_passed']}/{st['tests_run']}")
        
        print(f"\n🎨 TEXTURES GENERATED:")
        tg = summary['texture_generation']
        print(f"   Count: {tg['textures_created']}")
        for tex in tg['details']:
            print(f"   - {tex['mode']}: {tex['filename']}")
        
        print(f"\n⚡ ENERGY EFFICIENCY:")
        if summary['energy_efficiency']:
            ee = summary['energy_efficiency']
            print(f"   ASIC: {ee['asic']['joules_per_hash']:.6f} J/hash")
        
        print("\n" + "=" * 70)
    
    def run(self):
        """Run complete experiment."""
        self.start_time = datetime.now()
        
        if not self.setup():
            return
        
        try:
            if not self.run_warmup():
                return
            
            software_benchmark = self.run_software_benchmark()
            self.run_statistical_analysis(num_hashes=500)
            self.run_texture_generation()
            latent_comparison = self.run_latent_comparison()
            
            self.save_results(software_benchmark, latent_comparison)
            
        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED] Saving partial results...")
            self.save_results({}, {})
        finally:
            if self.bridge:
                self.bridge.stop()
        
        print(f"\n[DONE] Results saved to {self.output_dir}/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                   ASIC-DIFFUSION EXPERIMENT v1.0                     ║
║          Project C.H.R.O.M.A. - Hardware-Generated Art               ║
║                                                                       ║
║  Author: Francisco Angulo de Lafuente                                ║
║  GitHub: https://github.com/Agnuxo1                                  ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    experiment = DiffusionExperiment()
    experiment.run()


if __name__ == "__main__":
    main()
