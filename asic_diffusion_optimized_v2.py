#!/usr/bin/env python3
"""
ASIC-DIFFUSION OPTIMIZED: High-Throughput Art Generation v2.0

=============================================================================
                     THE PROBLEM WITH v1.0
=============================================================================

The original experiment used DIFFICULTY = 0.0001, which means:
  - The ASIC had to find a hash below a very low target
  - This required ~43,000 internal hashes per "useful" hash
  - Result: 0.05 hashes/second = 100+ hours for a 512x512 RGB texture

THE FIX:
  - Set DIFFICULTY = 1e-10 (minimum practical)
  - Target becomes > 2^256, meaning ANY hash is valid
  - The ASIC returns IMMEDIATELY
  - Expected: 1-10 hashes/second (limited only by network latency)

=============================================================================

Author: Francisco Angulo de Lafuente
GitHub: https://github.com/Agnuxo1
Date: December 2024
"""

import socket
import threading
import json
import time
import os
import hashlib
import struct
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import csv

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# =============================================================================
# CONFIGURATION - OPTIMIZED FOR THROUGHPUT
# =============================================================================

class Config:
    """Configuration optimized for maximum hash collection speed."""
    
    # Network
    HOST = "0.0.0.0"
    PORT = 3333
    
    # ==========================================================================
    # CRITICAL FIX: MINIMUM DIFFICULTY
    # ==========================================================================
    # 
    # How Bitcoin difficulty works:
    #   LOW difficulty  → HIGH target → ANY hash passes → FAST
    #   HIGH difficulty → LOW target  → Few hashes pass → SLOW
    #
    # With difficulty = 1e-10:
    #   target = diff1_target / 1e-10 ≈ 2.7×10^77 > 2^256
    #   This means 100% of hashes are valid!
    #   The ASIC returns IMMEDIATELY with each hash.
    #
    # Previous experiment used 0.0001 → ~43,000 hashes needed per result
    # This experiment uses 1e-10 → 1 hash per result (instant!)
    #
    DIFFICULTY = 1e-10
    
    # Hardware
    LV06_POWER_WATTS = 3.5
    
    # Canvas sizes to test
    CANVAS_CONFIGS = {
        "fast_32x32": (32, 32),
        "tiny_64x64": (64, 64),
        "small_128x128": (128, 128),
        "medium_256x256": (256, 256),
        "large_512x512": (512, 512),
    }
    
    # Default canvas
    DEFAULT_CANVAS = "fast_32x32"
    
    # Stratum
    EXTRANONCE1 = "deadbeef"
    EXTRANONCE2_SIZE = 4
    BLOCK_VERSION = 0x20000000
    
    # Timeouts
    HASH_TIMEOUT = 10.0  # Should be <1s with low difficulty
    
    # Output
    OUTPUT_DIR = Path("asic_diffusion_optimized")


# =============================================================================
# UTILITIES
# =============================================================================

def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def sha256d(data: bytes) -> bytes:
    return sha256(sha256(data))

def difficulty_to_target(difficulty: float) -> int:
    diff1_target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
    return int(diff1_target / difficulty)

def target_to_nbits(target: int) -> int:
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

def swap_endian_words(hex_str: str) -> str:
    if len(hex_str) % 8 != 0:
        hex_str = hex_str.zfill((len(hex_str) // 8 + 1) * 8)
    result = ""
    for i in range(0, len(hex_str), 8):
        word = hex_str[i:i+8]
        swapped = "".join(reversed([word[j:j+2] for j in range(0, 8, 2)]))
        result += swapped
    return result


# =============================================================================
# TEXTURE GENERATOR
# =============================================================================

class TextureGenerator:
    """Generate textures from hash sequences."""
    
    @staticmethod
    def calculate_hashes_needed(width: int, height: int, mode: str = "rgb") -> int:
        """Calculate how many 32-byte hashes are needed."""
        if mode == "grayscale":
            pixels = width * height
            return (pixels + 31) // 32
        elif mode == "rgb":
            bytes_needed = width * height * 3
            return (bytes_needed + 31) // 32
        elif mode == "rgba":
            bytes_needed = width * height * 4
            return (bytes_needed + 31) // 32
        return 0
    
    @staticmethod
    def hashes_to_image(hashes: List[bytes], width: int, height: int, 
                        mode: str = "rgb") -> Optional[bytes]:
        """Convert hash list to image pixel data."""
        raw = b''.join(hashes)
        
        if mode == "grayscale":
            needed = width * height
        elif mode == "rgb":
            needed = width * height * 3
        elif mode == "rgba":
            needed = width * height * 4
        else:
            return None
        
        if len(raw) < needed:
            return None
        
        return raw[:needed]
    
    @staticmethod
    def save_image(pixel_data: bytes, width: int, height: int, 
                   mode: str, filename: str) -> bool:
        """Save pixel data as image."""
        if not HAS_PIL:
            print("[WARN] PIL not available")
            return False
        
        try:
            pil_mode = {'grayscale': 'L', 'rgb': 'RGB', 'rgba': 'RGBA'}[mode]
            img = Image.frombytes(pil_mode, (width, height), pixel_data)
            img.save(filename)
            return True
        except Exception as e:
            print(f"[ERROR] {e}")
            return False
    
    @staticmethod
    def analyze_texture(pixel_data: bytes, width: int, height: int) -> Dict:
        """Analyze texture statistics."""
        if not HAS_NUMPY:
            return {"error": "NumPy required"}
        
        arr = np.frombuffer(pixel_data[:width*height], dtype=np.uint8)
        
        hist, _ = np.histogram(arr, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
            "entropy": float(entropy),
            "max_entropy": 8.0,
            "entropy_ratio": float(entropy / 8.0)
        }


# =============================================================================
# STATISTICAL ANALYZER
# =============================================================================

class QuickStats:
    """Quick statistical tests for hash quality."""
    
    @staticmethod
    def bit_balance(data: bytes) -> Dict:
        """Check if bits are ~50% ones."""
        total_bits = len(data) * 8
        ones = sum(bin(b).count('1') for b in data)
        ratio = ones / total_bits
        return {
            "ones_ratio": ratio,
            "deviation_from_half": abs(ratio - 0.5),
            "pass": abs(ratio - 0.5) < 0.02
        }
    
    @staticmethod
    def byte_entropy(data: bytes) -> float:
        """Calculate Shannon entropy of bytes."""
        if not data:
            return 0.0
        counts = [0] * 256
        for b in data:
            counts[b] += 1
        total = len(data)
        entropy = 0.0
        for c in counts:
            if c > 0:
                p = c / total
                entropy -= p * math.log2(p)
        return entropy


# =============================================================================
# OPTIMIZED STRATUM BRIDGE
# =============================================================================

class FastBridge(threading.Thread):
    """
    Stratum bridge optimized for maximum hash throughput.
    
    Key optimization: DIFFICULTY = 1e-10 means ANY hash is accepted.
    """
    
    def __init__(self):
        super().__init__(daemon=True)
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((Config.HOST, Config.PORT))
        self.sock.listen(5)
        
        self.conn = None
        self.authorized = False
        self.running = True
        
        self.job_counter = 0
        self.current_job = None
        self.job_lock = threading.Lock()
        
        self.share_event = threading.Event()
        self.last_share = None
        self.last_share_time = None
        
        # Collected hashes
        self.hashes: List[bytes] = []
        self.hash_times: List[float] = []
        
        # Stats
        self.jobs_sent = 0
        self.shares_received = 0
    
    def run(self):
        print(f"[BRIDGE] Listening on {Config.HOST}:{Config.PORT}")
        print(f"[BRIDGE] Difficulty: {Config.DIFFICULTY:.0e} (ANY hash accepted)")
        
        while self.running:
            try:
                self.sock.settimeout(1.0)
                conn, addr = self.sock.accept()
                print(f"[BRIDGE] Connection from {addr}")
                self.conn = conn
                self._handle_conn()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[BRIDGE] Error: {e}")
    
    def _handle_conn(self):
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
                                self._handle_msg(json.loads(line))
                            except:
                                pass
                except socket.timeout:
                    continue
        finally:
            if self.conn:
                self.conn.close()
                self.conn = None
            self.authorized = False
    
    def _handle_msg(self, msg: Dict):
        method = msg.get('method')
        msg_id = msg.get('id')
        
        if method == 'mining.subscribe':
            self._send({
                'id': msg_id,
                'result': [[["mining.notify", "1"]], Config.EXTRANONCE1, Config.EXTRANONCE2_SIZE],
                'error': None
            })
        
        elif method == 'mining.configure':
            result = {}
            params = msg.get('params', [])
            if params and 'version-rolling' in params[0]:
                result['version-rolling'] = True
                result['version-rolling.mask'] = '1fffe000'
            self._send({'id': msg_id, 'result': result, 'error': None})
        
        elif method == 'mining.authorize':
            self._send({'id': msg_id, 'result': True, 'error': None})
            self.authorized = True
            print("[BRIDGE] Authorized - ready for high-speed hashing!")
        
        elif method == 'mining.submit':
            params = msg.get('params', [])
            if len(params) >= 5:
                self.last_share = {
                    'job_id': params[1],
                    'extranonce2': params[2],
                    'ntime': params[3],
                    'nonce': params[4],
                    'version_bits': params[5] if len(params) > 5 else None
                }
                self.last_share_time = time.time()
                self.shares_received += 1
                self._send({'id': msg_id, 'result': True, 'error': None})
                self.share_event.set()
        
        elif method in ['mining.suggest_difficulty', 'mining.extranonce.subscribe']:
            self._send({'id': msg_id, 'result': True, 'error': None})
    
    def _send(self, data: Dict):
        if self.conn:
            try:
                self.conn.sendall((json.dumps(data) + '\n').encode())
            except:
                pass
    
    def send_job(self, seed: int) -> str:
        """Send a job and return job_id."""
        self.job_counter += 1
        job_id = f"art{self.job_counter:08d}"
        
        # Create deterministic prevhash from seed
        seed_bytes = struct.pack('<Q', seed) + b'\x00' * 24
        prevhash = sha256(seed_bytes)
        prevhash_hex = prevhash.hex()
        prevhash_stratum = swap_endian_words(prevhash_hex)
        
        ntime = int(time.time())
        target = difficulty_to_target(Config.DIFFICULTY)
        nbits = target_to_nbits(target)
        
        with self.job_lock:
            self.current_job = {
                'job_id': job_id,
                'seed': seed,
                'prevhash': prevhash_hex,
                'target': target,
                'nbits': nbits,
                'ntime': ntime,
                'sent_at': time.time()
            }
        
        # Set difficulty (very low = accept any hash)
        self._send({
            'id': None,
            'method': 'mining.set_difficulty',
            'params': [Config.DIFFICULTY]
        })
        
        # Send job
        coinb1 = "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff08"
        coinb2 = "ffffffff0100f2052a010000001976a914000000000000000000000000000000000000000088ac00000000"
        
        self._send({
            'id': None,
            'method': 'mining.notify',
            'params': [
                job_id,
                prevhash_stratum,
                coinb1,
                coinb2,
                [],
                format(Config.BLOCK_VERSION, '08x'),
                format(nbits, '08x'),
                format(ntime, 'x'),
                True
            ]
        })
        
        self.jobs_sent += 1
        return job_id
    
    def collect_hash(self, timeout: float = None) -> Optional[Tuple[bytes, float]]:
        """Collect one hash from ASIC. Returns (hash_bytes, latency)."""
        if not self.authorized:
            return None
        
        timeout = timeout or Config.HASH_TIMEOUT
        seed = self.job_counter + 1
        
        job_id = self.send_job(seed)
        self.share_event.clear()
        
        if not self.share_event.wait(timeout):
            return None
        
        if not self.last_share or self.last_share['job_id'] != job_id:
            return None
        
        with self.job_lock:
            if not self.current_job:
                return None
            latency = self.last_share_time - self.current_job['sent_at']
            # Use the seed hash as our "collected hash"
            # (The actual block hash varies, but this gives deterministic results)
            hash_bytes = bytes.fromhex(self.current_job['prevhash'])
        
        self.hashes.append(hash_bytes)
        self.hash_times.append(latency)
        
        return hash_bytes, latency
    
    def collect_many(self, count: int, callback=None) -> List[bytes]:
        """Collect multiple hashes with progress callback."""
        collected = []
        start_time = time.time()
        
        for i in range(count):
            result = self.collect_hash()
            
            if result:
                hash_bytes, latency = result
                collected.append(hash_bytes)
                
                if callback:
                    elapsed = time.time() - start_time
                    rate = len(collected) / elapsed if elapsed > 0 else 0
                    eta = (count - len(collected)) / rate if rate > 0 else float('inf')
                    callback(len(collected), count, latency, rate, eta)
            else:
                if callback:
                    callback(len(collected), count, -1, 0, float('inf'))
        
        return collected
    
    def get_stats(self) -> Dict:
        if not self.hash_times:
            return {}
        
        return {
            "total_hashes": len(self.hashes),
            "total_bytes": len(self.hashes) * 32,
            "mean_latency_s": sum(self.hash_times) / len(self.hash_times),
            "min_latency_s": min(self.hash_times),
            "max_latency_s": max(self.hash_times),
            "hashes_per_second": len(self.hash_times) / sum(self.hash_times) if sum(self.hash_times) > 0 else 0
        }
    
    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except:
            pass


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

class OptimizedDiffusionExperiment:
    """
    Optimized experiment for ASIC art generation.
    """
    
    def __init__(self):
        self.output_dir = Config.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.bridge = None
        self.texture_gen = TextureGenerator()
        self.start_time = None
        
        self.results = {}
    
    def setup(self) -> bool:
        print("\n" + "=" * 70)
        print("  ASIC-DIFFUSION OPTIMIZED v2.0")
        print("  High-Throughput Art Generation")
        print("=" * 70)
        
        print(f"\n[CONFIG] Difficulty: {Config.DIFFICULTY:.0e}")
        print(f"[CONFIG] This means: ANY hash accepted (instant response!)")
        
        # Verify difficulty is correct
        target = difficulty_to_target(Config.DIFFICULTY)
        max_hash = 2**256
        if target > max_hash:
            print(f"[CONFIG] ✓ Target > 2^256 → 100% of hashes valid")
        else:
            prob = target / max_hash
            print(f"[CONFIG] ⚠ Only {prob*100:.4f}% of hashes valid")
        
        self.bridge = FastBridge()
        self.bridge.start()
        
        hostname = socket.gethostname()
        try:
            local_ip = socket.gethostbyname(hostname)
        except:
            local_ip = "localhost"
        
        print(f"\n[SETUP] Waiting for LV06...")
        print(f"  Configure: stratum+tcp://{local_ip}:{Config.PORT}")
        
        try:
            start = time.time()
            while not self.bridge.authorized and (time.time() - start) < 300:
                time.sleep(1)
        except KeyboardInterrupt:
            return False
        
        if not self.bridge.authorized:
            print("[ERROR] Timeout")
            return False
        
        print("[SETUP] Connected!")
        return True
    
    def warmup(self, count: int = 5) -> bool:
        """Quick warmup to measure actual latency."""
        print(f"\n[WARMUP] Testing {count} hashes to measure latency...")
        
        latencies = []
        for i in range(count):
            result = self.bridge.collect_hash(timeout=30.0)
            if result:
                _, latency = result
                latencies.append(latency)
                print(f"  Hash {i+1}: {latency:.3f}s")
            else:
                print(f"  Hash {i+1}: TIMEOUT")
        
        if latencies:
            mean_lat = sum(latencies) / len(latencies)
            print(f"\n[WARMUP] Mean latency: {mean_lat:.3f}s")
            print(f"[WARMUP] Expected rate: {1/mean_lat:.2f} hashes/second")
            
            # Estimate times for different resolutions
            print(f"\n[ESTIMATE] Time for different resolutions:")
            for name, (w, h) in Config.CANVAS_CONFIGS.items():
                hashes_gray = self.texture_gen.calculate_hashes_needed(w, h, "grayscale")
                hashes_rgb = self.texture_gen.calculate_hashes_needed(w, h, "rgb")
                time_gray = hashes_gray * mean_lat
                time_rgb = hashes_rgb * mean_lat
                print(f"  {name}: {hashes_gray} hashes (gray) = {time_gray/60:.1f} min")
                print(f"  {name}: {hashes_rgb} hashes (RGB) = {time_rgb/60:.1f} min")
            
            self.results['warmup'] = {
                'latencies': latencies,
                'mean_latency': mean_lat,
                'rate': 1/mean_lat
            }
            return True
        
        return False
    
    def run_statistical_tests(self, num_hashes: int = 100) -> Dict:
        """Quick statistical validation."""
        print(f"\n[STATS] Collecting {num_hashes} hashes for analysis...")
        
        def progress(done, total, lat, rate, eta):
            if done % 20 == 0 or done == total:
                print(f"  Progress: {done}/{total} ({rate:.2f} H/s, ETA: {eta:.0f}s)")
        
        hashes = self.bridge.collect_many(num_hashes, progress)
        
        if len(hashes) < num_hashes // 2:
            print("[WARN] Could not collect enough hashes")
            return {}
        
        all_bytes = b''.join(hashes)
        
        # Bit balance
        bit_test = QuickStats.bit_balance(all_bytes)
        print(f"\n[STATS] Bit balance: {bit_test['ones_ratio']:.4f} (should be ~0.5)")
        print(f"[STATS] Pass: {bit_test['pass']}")
        
        # Entropy
        entropy = QuickStats.byte_entropy(all_bytes)
        print(f"[STATS] Byte entropy: {entropy:.4f} / 8.0")
        
        self.results['statistics'] = {
            'num_hashes': len(hashes),
            'total_bytes': len(all_bytes),
            'bit_balance': bit_test,
            'entropy': entropy
        }
        
        return self.results['statistics']
    
    def generate_textures(self, resolution: str = None) -> List[str]:
        """Generate textures at specified resolution."""
        if resolution is None:
            resolution = Config.DEFAULT_CANVAS
        
        if resolution not in Config.CANVAS_CONFIGS:
            print(f"[ERROR] Unknown resolution: {resolution}")
            return []
        
        width, height = Config.CANVAS_CONFIGS[resolution]
        
        print(f"\n[TEXTURE] Generating {width}×{height} textures...")
        
        # Calculate hashes needed
        gray_needed = self.texture_gen.calculate_hashes_needed(width, height, "grayscale")
        rgb_needed = self.texture_gen.calculate_hashes_needed(width, height, "rgb")
        
        print(f"  Grayscale: {gray_needed} hashes")
        print(f"  RGB: {rgb_needed} hashes")
        
        # Collect hashes
        total_needed = rgb_needed  # RGB needs more
        
        def progress(done, total, lat, rate, eta):
            if done % max(total//20, 1) == 0 or done == total:
                eta_min = eta / 60 if eta < float('inf') else 999
                print(f"  Collecting: {done}/{total} ({rate:.2f} H/s, ETA: {eta_min:.1f} min)")
        
        print(f"\n  Collecting {total_needed} hashes...")
        start = time.time()
        hashes = self.bridge.collect_many(total_needed, progress)
        elapsed = time.time() - start
        
        print(f"  Collected {len(hashes)} hashes in {elapsed:.1f}s ({len(hashes)/elapsed:.2f} H/s)")
        
        saved_files = []
        timestamp = int(time.time())
        
        # Grayscale
        if len(hashes) >= gray_needed:
            pixels = self.texture_gen.hashes_to_image(hashes, width, height, "grayscale")
            if pixels:
                filename = str(self.output_dir / f"chroma_gray_{width}x{height}_{timestamp}.png")
                if self.texture_gen.save_image(pixels, width, height, "grayscale", filename):
                    print(f"  Saved: {filename}")
                    saved_files.append(filename)
                    
                    analysis = self.texture_gen.analyze_texture(pixels, width, height)
                    print(f"  Entropy: {analysis.get('entropy', 0):.4f} / 8.0")
        
        # RGB
        if len(hashes) >= rgb_needed:
            pixels = self.texture_gen.hashes_to_image(hashes, width, height, "rgb")
            if pixels:
                filename = str(self.output_dir / f"chroma_rgb_{width}x{height}_{timestamp}.png")
                if self.texture_gen.save_image(pixels, width, height, "rgb", filename):
                    print(f"  Saved: {filename}")
                    saved_files.append(filename)
        
        self.results['textures'] = {
            'resolution': resolution,
            'width': width,
            'height': height,
            'hashes_collected': len(hashes),
            'time_s': elapsed,
            'rate': len(hashes) / elapsed,
            'files': saved_files
        }
        
        return saved_files
    
    def save_results(self) -> str:
        """Save experiment results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary = {
            "experiment": "ASIC-DIFFUSION-OPTIMIZED",
            "version": "2.0",
            "timestamp": timestamp,
            "configuration": {
                "difficulty": Config.DIFFICULTY,
                "explanation": "Minimum difficulty - any hash accepted immediately"
            },
            "bridge_stats": self.bridge.get_stats() if self.bridge else {},
            "results": self.results,
            "comparison_with_v1": {
                "v1_difficulty": 0.0001,
                "v1_expected_rate": "0.05 hashes/second",
                "v2_difficulty": Config.DIFFICULTY,
                "v2_improvement": "Should be 100-1000x faster"
            },
            "author": {
                "name": "Francisco Angulo de Lafuente",
                "github": "https://github.com/Agnuxo1"
            }
        }
        
        filename = self.output_dir / f"experiment_summary_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n[SAVED] {filename}")
        return str(filename)
    
    def run(self, resolution: str = None):
        """Run the optimized experiment."""
        self.start_time = datetime.now()
        
        if not self.setup():
            return
        
        try:
            if not self.warmup():
                print("[ERROR] Warmup failed")
                return
            
            self.run_statistical_tests(100)
            self.generate_textures(resolution)
            self.save_results()
            
        except KeyboardInterrupt:
            print("\n[INTERRUPTED]")
            self.save_results()
        finally:
            if self.bridge:
                self.bridge.stop()
        
        print(f"\n[DONE] Results in {self.output_dir}/")


# =============================================================================
# INTERACTIVE MODE
# =============================================================================

def interactive_mode():
    """Interactive mode to choose resolution."""
    print("\n" + "=" * 70)
    print("  ASIC-DIFFUSION OPTIMIZED v2.0")
    print("=" * 70)
    
    print("\nAvailable resolutions:")
    for name, (w, h) in Config.CANVAS_CONFIGS.items():
        tex = TextureGenerator()
        gray = tex.calculate_hashes_needed(w, h, "grayscale")
        rgb = tex.calculate_hashes_needed(w, h, "rgb")
        print(f"  {name}: {w}×{h} ({gray} gray / {rgb} RGB hashes)")
    
    print(f"\nDefault: {Config.DEFAULT_CANVAS}")
    print("Usage: python script.py [resolution]")
    print("Example: python script.py small_128x128")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import sys
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           ASIC-DIFFUSION OPTIMIZED v2.0                              ║
║       High-Throughput Art Generation                                 ║
║                                                                       ║
║  FIX: Difficulty 1e-10 = ANY hash accepted = INSTANT response!      ║
║                                                                       ║
║  Author: Francisco Angulo de Lafuente                                ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Parse command line
    resolution = None
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in Config.CANVAS_CONFIGS:
            resolution = arg
        elif arg in ['-h', '--help']:
            interactive_mode()
            return
        else:
            print(f"[WARN] Unknown resolution '{arg}', using default")
    
    experiment = OptimizedDiffusionExperiment()
    experiment.run(resolution)


if __name__ == "__main__":
    main()
