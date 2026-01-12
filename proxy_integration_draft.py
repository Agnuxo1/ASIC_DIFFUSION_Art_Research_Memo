#!/usr/bin/env python3
"""
S9 ART BRIDGE v15.0 (Proxy Legacy Integration)
==============================================
Uses the verified StratumProxy class from config.py.
Integrates with Silicon TV renderer.
"""

import sys
import os
import threading
import json
import time

# Add root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import S9Config, StratumProxy

class SiliconTVBridge:
    def __init__(self):
        self.config = S9Config()
        self.config.S9_IP = "192.168.0.16"
        self.config.STRATUM_PORT = 3333
        self.config.D_BASE = 1.0
        
        self.proxy = StratumProxy(self.config)
        self.running = True
        
    def run_stratum(self):
        self.proxy.start()
        while self.running:
            if self.proxy.wait_for_connection():
                while self.proxy.running:
                    try:
                        data = self.proxy.client_conn.recv(4096)
                        if not data: break
                        responses = self.proxy.process_message(data)
                        for resp in responses:
                            self.proxy.client_conn.send(resp.encode())
                        
                        # Periodically send new jobs to keep it noisy
                        if time.time() - self.proxy.last_job_time > 5:
                            self.proxy.send_job(u_value=0.5)
                            self.proxy.last_job_time = time.time()
                            
                    except Exception as e:
                        print(f"[RE-CONNECT] {e}")
                        break
            time.sleep(1)

    def run_api_server(self):
        import socket
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", 4000))
        server.listen(5)
        while self.running:
            try:
                client, _ = server.accept()
                data = client.recv(1024).decode()
                req = json.loads(data)
                if req.get('type') == 'get_entropy':
                    count = int(req.get('count', 1000))
                    # Extract raw entropy from proxy.shares
                    # (Note: StratumProxy.record_share only logs timestamp/job_id.
                    # We need the RAW HEX for the Silicon TV.
                    # I will patch StratumProxy.process_message to capture hex).
                    pass
            except: pass

    def start(self):
        # We need a small patch to the proxy to capture the raw hex nonces
        t_stratum = threading.Thread(target=self.run_stratum, daemon=True)
        t_stratum.start()
        
        while self.running:
            time.sleep(10)
            shares = self.proxy.shares
            print(f"[TV-STATS] Raw Shares Captured: {len(shares)}")

if __name__ == "__main__":
    # Integration logic needs more than just class usage.
    # I will stick to my standalone bridge but with EXACT logic from config.py
    pass
