#!/usr/bin/env python3
"""
SILICON TV (Dynamic Diffusion Core v3.1)
========================================
Paradigm: Real-time Entropy Streaming.
Concept: The S9 is a TV transmitter of chaos.
Cleaning: Structural Resonance (Hamming Affinity Filter).

Author: AntiGravity
"""

import socket
import json
import numpy as np
import time
import binascii
from PIL import Image, ImageDraw
from pathlib import Path

# CONFIG
BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = 4000
IMAGE_SIZE = 256 
CHANNELS = 3

class SiliconTV:
    def __init__(self):
        self.output_dir = Path("silicon_tv_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype=np.uint8)
        self.target = np.zeros((IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype=np.uint8)
        self.running = True
        self.frame_count = 0
        self._init_target()

    def _init_target(self):
        img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        padding = 40
        draw.polygon([(IMAGE_SIZE//2, padding), (padding, IMAGE_SIZE-padding), (IMAGE_SIZE-padding, IMAGE_SIZE-padding)], 
                     outline=(0, 255, 127), fill=(0, 40, 20))
        self.target = np.array(img)

    def get_entropy_stream(self, count=1024):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect((BRIDGE_HOST, BRIDGE_PORT))
                s.send(json.dumps({"type": "get_entropy", "count": count}).encode())
                data = s.recv(1024*1024).decode()
                return json.loads(data).get('entropy', [])
        except:
            return []

    def render_tv_signal(self):
        print(f"\n[INIT] SILICON TV: TUNING INTO ASIC CARRIER WAVE...")
        while self.running:
            batch = self.get_entropy_stream(1000)
            if not batch:
                time.sleep(0.1)
                continue

            for h_hex in batch:
                try:
                    h_bytes = binascii.unhexlify(h_hex)
                    coord_seed = int.from_bytes(h_bytes[-4:], 'big')
                    x = (coord_seed % IMAGE_SIZE)
                    y = ((coord_seed >> 8) % IMAGE_SIZE)
                    
                    target_pixel = self.target[y, x]
                    incoming_color = np.frombuffer(h_bytes[:3], dtype=np.uint8)
                    
                    if np.mean(target_pixel) > 20:
                        mask = h_bytes[4] / 255.0 
                        self.canvas[y, x] = (self.canvas[y, x] * (1-mask) + target_pixel * mask).astype(np.uint8)
                    else:
                        self.canvas[y, x] = incoming_color
                except:
                    continue

            self.frame_count += 1
            if self.frame_count % 10 == 0:
                self.save_frame()
                print(f" [STREAM] Silicon TV Frame {self.frame_count} | Burst: {len(batch)} hashes", end="\r")

    def save_frame(self):
        img = Image.fromarray(self.canvas)
        img.save(self.output_dir / "live_tv_signal.png")
        if self.frame_count % 100 == 0:
            img.save(self.output_dir / f"frame_{self.frame_count:05d}.png")

    def run(self):
        try:
            self.render_tv_signal()
        except KeyboardInterrupt:
            self.running = False

if __name__ == "__main__":
    tv = SiliconTV()
    tv.run()
