#!/usr/bin/env python3
"""
720p COMPARATIVE BENCHMARK: ASIC S9 vs CPU (V3 - DEFINITIVE)
===========================================================
"""
import os
import sys
import time
import json
import socket
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

WIDTH, HEIGHT = 1280, 720
BENCHMARK_SHARES = 100
S9_POWER_W = 1300.0
CPU_POWER_W = 95.0
S9_TH_S = 14.6
CPU_MH_S = 2.0

def calculate_entropy(data_hex_list):
    all_bytes = b"".join([bytes.fromhex(h) for h in data_hex_list])
    if not all_bytes: return 0
    counts = np.bincount(np.frombuffer(all_bytes, dtype=np.uint8), minlength=256)
    probs = counts / len(all_bytes)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def run_cpu():
    print("[BENCH] Running CPU (720p)...")
    data_samples = []
    t0 = time.perf_counter()
    for i in range(BENCHMARK_SHARES):
        _ = np.random.random((HEIGHT, WIDTH)).astype(np.float32)
        data_samples.append(os.urandom(32).hex())
    t1 = time.perf_counter()
    duration = t1 - t0
    
    h_j = (CPU_MH_S * 1e6) / CPU_POWER_W
    return {
        "Device": "CPU (i7-10700K Approx)",
        "Interface FPS": BENCHMARK_SHARES / duration,
        "Entropy (bits)": calculate_entropy(data_samples),
        "Internal Hash/Joule": h_j,
        "Efficiency Factor": 1.0,
        "Paradigm": "Pseudo-Random"
    }

def run_asic():
    print("[BENCH] Running ASIC (720p)...")
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
            data_samples.append(res.get("nonce", "00000000"))
            # HD Matrix overhead simulation
            _ = np.random.random((HEIGHT, WIDTH)).astype(np.float32)
            
        t1_total = time.perf_counter()
        duration = t1_total - t0_total
        
        h_j_asic = (S9_TH_S * 1e12) / S9_POWER_W
        h_j_cpu = (CPU_MH_S * 1e6) / CPU_POWER_W
        return {
            "Device": "ASIC (Antminer S9)",
            "Interface FPS": BENCHMARK_SHARES / duration,
            "Entropy (bits)": calculate_entropy(data_samples),
            "Internal Hash/Joule": h_j_asic,
            "Efficiency Factor": h_j_asic / h_j_cpu,
            "Paradigm": "Hardware Physics"
        }
    except Exception as e:
        print(f"[ERROR] ASIC: {e}")
        return None

if __name__ == "__main__":
    print("-" * 40)
    print("720p HD SILICON BENCHMARK V3")
    print("-" * 40)
    results = [run_cpu(), run_asic()]
    df = pd.DataFrame([r for r in results if r])
    df.to_csv("D:/ASIC-ANTMINER_S9/ASIC_DIFFUSION_Art_Research_Memo/comparison_720p_v3.csv", index=False)
    print("\n[COMPLETE]")
    print(df.to_string(index=False))
