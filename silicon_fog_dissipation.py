#!/usr/bin/env python3
"""
SILICON FOG DISSIPATION ENGINE
==============================
El ASIC no crea la imagen - el ASIC DISIPA LA NIEBLA.

PARADIGMA:
---------
1. Creamos una "niebla lógica estructurada" (target image + ruido)
2. El timing jitter del ASIC actúa como filtro de interferencia
3. Donde hay RESONANCIA, la niebla se disipa y la imagen emerge
4. Es como holografía: la imagen está codificada, el ASIC la revela

Autor: Francisco Angulo de Lafuente
Fecha: Enero 2026

"El silicio habla. Escuchamos. La niebla se disipa."
"""

import os
import sys
import time
import numpy as np
from PIL import Image, ImageDraw
from collections import deque
from datetime import datetime

# Add parent for StratumProxy import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 's9_experiments', 's9_experiments'))
from config import S9Config, StratumProxy

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CANVAS_SIZE = 128  # Smaller for high-speed TV effect
OUTPUT_DIR = os.path.dirname(__file__)
TARGET_ART = "triangle"  # What image emerges from the fog

# ==============================================================================
# FOG ENGINE
# ==============================================================================

class SiliconFogEngine:
    """
    Motor de Disipación de Niebla por Resonancia de Silicio.
    
    La niebla contiene la estructura del arte.
    El timing del ASIC "limpia" la niebla donde hay resonancia.
    """
    
    def __init__(self, canvas_size: int = CANVAS_SIZE):
        self.size = canvas_size
        self.fog = None  # La niebla actual
        self.target = None  # La imagen objetivo codificada
        self.frame_count = 0
        self.timing_history = deque(maxlen=1000)
        
        # Initialize
        self.create_target_art()
        self.initialize_fog()
        
    def create_target_art(self):
        """Crea la imagen artística que emerge de la niebla."""
        img = Image.new('L', (self.size, self.size), 0)
        draw = ImageDraw.Draw(img)
        
        if TARGET_ART == "triangle":
            # Triángulo centrado
            cx, cy = self.size // 2, self.size // 2
            r = self.size // 3
            points = [
                (cx, cy - r),  # Top
                (cx - r, cy + r),  # Bottom left
                (cx + r, cy + r),  # Bottom right
            ]
            draw.polygon(points, fill=255)
        elif TARGET_ART == "circle":
            margin = self.size // 4
            draw.ellipse([margin, margin, self.size - margin, self.size - margin], fill=255)
        elif TARGET_ART == "square":
            margin = self.size // 4
            draw.rectangle([margin, margin, self.size - margin, self.size - margin], fill=255)
        
        self.target = np.array(img, dtype=np.float32) / 255.0
        print(f"[FOG] Target art '{TARGET_ART}' created: {self.size}x{self.size}")
        
    def initialize_fog(self):
        """Inicializa la niebla con ruido estructurado."""
        # La niebla es ruido, pero CONTIENE la estructura del target
        # (codificada como variaciones sutiles de frecuencia)
        noise = np.random.random((self.size, self.size)).astype(np.float32)
        
        # Mezcla: fog = noise + subtle_target_structure
        # El target está "escondido" en la niebla
        self.fog = 0.8 * noise + 0.2 * self.target
        
        print(f"[FOG] Structured fog initialized. Target hidden within noise.")
        
    def dissipate_with_timing(self, delta_ms: float):
        """
        Usa el timing jitter del ASIC para disipar la niebla.
        
        El timing actúa como un "filtro de interferencia":
        - Timing bajo (rápido) = alta energía = más disipación
        - Timing alto (lento) = baja energía = menos disipación
        
        La disipación es SELECTIVA: afecta más donde hay estructura.
        """
        self.timing_history.append(delta_ms)
        
        # Normalizar timing a un valor de "energía de disipación"
        # Inverse: low timing = high energy
        max_delta = 500.0  # ms, cap for normalization
        energy = 1.0 - min(delta_ms / max_delta, 1.0)
        
        # Calcular el "patrón de resonancia" del timing actual
        # Usamos el timing para generar un patrón pseudo-aleatorio
        # pero DETERMINISTICO (mismo timing = mismo patrón)
        np.random.seed(int(delta_ms * 1000) % (2**31))
        resonance_pattern = np.random.random((self.size, self.size)).astype(np.float32)
        
        # La disipación es más fuerte donde:
        # 1. Hay alta energía (timing bajo)
        # 2. El patrón de resonancia "coincide" con la estructura del target
        
        # Calcular similitud local entre resonance_pattern y target
        similarity = 1.0 - np.abs(resonance_pattern - self.target)
        
        # Aplicar disipación: fog se mueve hacia target
        dissipation_strength = energy * similarity * 0.02  # Factor de ajuste
        self.fog = self.fog + dissipation_strength * (self.target - self.fog)
        
        # Clamp
        self.fog = np.clip(self.fog, 0, 1)
        
        self.frame_count += 1
        
    def get_current_image(self) -> Image.Image:
        """Retorna la imagen actual (fog en proceso de disipación)."""
        # Convertir fog a RGB
        gray = (self.fog * 255).astype(np.uint8)
        return Image.fromarray(gray, mode='L').convert('RGB')
    
    def save_frame(self, path: str = None):
        """Guarda el frame actual."""
        if path is None:
            path = os.path.join(OUTPUT_DIR, "live_fog_signal.png")
        img = self.get_current_image()
        img.save(path)
        return path

# ==============================================================================
# MAIN: SILICON TV STREAMER
# ==============================================================================

def run_silicon_tv():
    """
    Ejecuta el "Silicon TV" - streaming de disipación de niebla.
    Conecta al S9 via StratumProxy probado.
    """
    print("=" * 60)
    print("SILICON FOG DISSIPATION ENGINE")
    print("=" * 60)
    print("The ASIC doesn't CREATE the image.")
    print("The ASIC DISSIPATES THE FOG to reveal it.")
    print()
    
    # Initialize
    config = S9Config()
    config.S9_IP = "192.168.0.16"
    config.STRATUM_PORT = 3333
    config.D_BASE = 1.0  # Same as working experiments
    
    engine = SiliconFogEngine(canvas_size=CANVAS_SIZE)
    proxy = StratumProxy(config)
    
    try:
        proxy.start()
        
        print(f"\n[WAITING] Expecting S9 connection on port {config.STRATUM_PORT}...")
        print(f"[NOTE] Make sure S9 pools are set to: stratum+tcp://192.168.0.11:{config.STRATUM_PORT}")
        
        if not proxy.wait_for_connection():
            print("[ERROR] No miner connection. Exiting.")
            return
        
        print("[CONNECTED] S9 attached. Beginning fog dissipation...")
        
        # Initial setup
        proxy.send_difficulty(config.D_BASE)
        proxy.send_job(0.25)
        
        last_share_time = time.perf_counter()
        start_time = time.time()
        total_shares = 0
        
        while True:
            try:
                proxy.client_conn.settimeout(0.1)
                data = proxy.client_conn.recv(4096)
                
                if data:
                    responses = proxy.process_message(data)
                    for resp in responses:
                        proxy.client_conn.send(resp.encode())
                    
                    # Check for new shares
                    current_shares = list(proxy.shares)
                    if len(current_shares) > total_shares:
                        for share in current_shares[total_shares:]:
                            now = share["timestamp"]
                            delta_ms = (now - last_share_time) * 1000
                            
                            # Use timing to dissipate fog
                            engine.dissipate_with_timing(delta_ms)
                            
                            last_share_time = now
                            total_shares += 1
                        
                        # Progress every 100 shares
                        if total_shares % 100 == 0:
                            elapsed = time.time() - start_time
                            fps = engine.frame_count / elapsed if elapsed > 0 else 0
                            print(f"[TV] Frame {engine.frame_count} | "
                                  f"Shares: {total_shares} | "
                                  f"FPS: {fps:.1f}")
                            
                            # Save current state
                            engine.save_frame()
                    
                    # Re-handshake detection
                    if b'mining.authorize' in data:
                        proxy.send_difficulty(config.D_BASE)
                        proxy.send_job(0.25)
                
            except Exception as e:
                # Reconnect on error
                if "timeout" not in str(e).lower():
                    print(f"[RECONNECT] {e}")
                    if proxy.wait_for_connection():
                        proxy.send_difficulty(config.D_BASE)
                        proxy.send_job(0.25)
                        last_share_time = time.perf_counter()
                continue
                
    except KeyboardInterrupt:
        print("\n[EXIT] Stopping Silicon TV...")
    finally:
        # Final save
        path = engine.save_frame()
        print(f"\n[SAVED] Final fog state: {path}")
        print(f"[STATS] Total frames: {engine.frame_count} | Total shares: {total_shares}")
        proxy.stop()

if __name__ == "__main__":
    run_silicon_tv()
