#!/usr/bin/env python3
"""
720p COMPARATIVE BENCHMARK: ASIC S9 vs CPU (V2)
==============================================
Metodología:
1. CPU: Genera ruido pseudo-aleatorio para disipar la niebla (1280x720).
2. ASIC: Usa jitter físico del S9 para disipar la niebla (1280x720).
3. Metricas: FPS, Eficiencia Energética (J/Op), Entropía de la señal, y H/J Puro.
"""

import os
import sys
import time
import json
import socket
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# configuration
WIDTH, HEIGHT = 1280, 720
BENCHMARK_SHARES = 200
S9_POWER_W = 1300.0
CPU_POWER_W = 95.0
S9_TH_S = 14.6
CPU_MH_S = 2.0 # Estimated for i7

def calculate_entropy(data_hex_list):
    all_bytes = b"".join([bytes.fromhex(h) for h in data_hex_list])
    if not all_bytes: return 0
    counts = np.bincount(np.frombuffer(all_bytes, dtype=np.uint8), minlength=256)
    probs = counts / len(all_bytes)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

class HDFogEngine:
    def __init__(self):
        self.fog = np.random.random((HEIGHT, WIDTH)).astype(np.float32)
        self.target = self._create_target()
        
    def _create_target(self):
        img = Image.new('L', (WIDTH, HEIGHT), 0)
        draw = ImageDraw.Draw(img)
        draw.polygon([(640, 100), (240, 620), (1040, 620)], fill=255)
        return np.array(img, dtype=np.float32) / 255.0

    def step(self, noise_pattern, energy=0.5):
        affinity = 1.0 - np.abs(noise_pattern - self.target)
        self.fog += 0.05 * energy * affinity * (self.target - self.fog)
        self.fog = np.clip(self.fog, 0, 1)

def run_cpu_bench():
    print("[BENCH] Starting CPU Run...")
    engine = HDFogEngine()
    data_samples = []
    t0 = time.perf_counter()
    for i in range(BENCHMARK_SHARES):
        noise = np.random.random((HEIGHT, WIDTH)).astype(np.float32)
        engine.step(noise)
        data_samples.append(os.urandom(32).hex())
    t1 = time.perf_counter()
    duration = t1 - t0
    
    h_j_cpu = (CPU_MH_S * 1e6) / CPU_POWER_W
    return {
        "Device": "CPU (i7-TDP)",
        "Interface FPS": BENCHMARK_SHARES / duration,
        "Entropy (bits)": calculate_entropy(data_samples),
        "Internal H/J": h_j_cpu,
        "Efficiency Gap": 1.0,
        "Note": "Pseudo-Random Logic"
    }

def run_asic_bench():
    print("[BENCH] Starting ASIC (S9) Run...")
    engine = HDFogEngine()
    data_samples = []
    try:
        t0_total = time.perf_counter()
        for i in range(BENCHMARK_SHARES):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect(("127.0.0.1", 4000))
            sock.send((json.dumps({"data": os.urandom(32).hex()}) + "\n").encode())
            resp = sock.recv(4096).decode()
            sock.close()
            if not resp: continue
            res = json.loads(resp)
            nonce = res.get("nonce", "0")
            np.random.seed(int(nonce, 16) % (2**31))
            noise = np.random.random((HEIGHT, WIDTH)).astype(np.float32)
            engine.step(noise, energy=1.0)
            data_samples.append(nonce)
        t1_total = time.perf_counter()
        duration = t1_total - t0_total
        
        h_j_asic = (S9_TH_S * 1e12) / S9_POWER_W
        h_j_cpu = (CPU_MH_S * 1e6) / CPU_POWER_W
        return {
            "Device": "ASIC (Antminer S9)",
            "Interface FPS": BENCHMARK_SHARES / duration,
            "Entropy (bits)": calculate_entropy(data_samples),
            "Internal H/J": h_j_asic,
            "Efficiency Gap": h_j_asic / h_j_cpu,
            "Note": "Hardware-Native Physics"
        }
    except Exception as e:
        print(f"[ERROR] ASIC Bench failed: {e}")
        return None

def main():
    print("=" * 60)
    print("COMPARATIVE BENCHMARK: 720p SILICON TV (V2)")
    print("=" * 60)
    results = [run_cpu_bench(), run_asic_bench()]
    results = [r for r in results if r]
    df = pd.DataFrame(results)
    csv_path = "D:/ASIC-ANTMINER_S9/ASIC_DIFFUSION_Art_Research_Memo/comparison_720p_asic_vs_cpu.csv"
    df.to_csv(csv_path, index=False)
    print("\n[RESULT TABLE]")
    print(df.to_string(index=False))
    print(f"\n[SAVED] Benchmark results: {csv_path}")

if __name__ == "__main__":
    main()
