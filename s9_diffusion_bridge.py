#!/usr/bin/env python3
"""
S9 ART BRIDGE v17.0 (Scientific Trace Mode)
===========================================
Verbose logging of every byte for protocol analysis.
Matches s9_experiments/config.py EXACTLY.
"""

import socket
import threading
import json
import time
import hashlib
import binascii
from collections import deque

# CONFIG
STRATUM_PORT = 3333
API_PORT = 4000
DIFFICULTY = 0.5 
BUFFER_SIZE = 500000

class ArtBridge:
    def __init__(self):
        self.running = True
        self.entropy_buffer = deque(maxlen=BUFFER_SIZE)
        self.buffer_lock = threading.Lock()
        self.workers = {}
        self.job_id = 0
        
    def start(self):
        threading.Thread(target=self.run_stratum_server, daemon=True).start()
        threading.Thread(target=self.run_api_server, daemon=True).start()
        print(f"[TRACE-BRIDGE-V17] LIVE. STRATUM:{STRATUM_PORT} | API:{API_PORT}")
        while self.running:
            time.sleep(5)
            with self.buffer_lock:
                print(f"[HEARTBEAT] Reservoir: {len(self.entropy_buffer)} | Workers: {len(self.workers)}")

    def run_stratum_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", STRATUM_PORT))
        server.listen(10)
        while self.running:
            try:
                conn, addr = server.accept()
                threading.Thread(target=self.handle_s9, args=(conn, addr), daemon=True).start()
            except: pass

    def handle_s9(self, conn, addr):
        worker_id = f"{addr[0]}:{addr[1]}"
        print(f"[CONN] {worker_id}")
        
        buffer = ""
        try:
            while self.running:
                data = conn.recv(65536)
                if not data: break
                
                # Raw Trace
                # print(f"[RECV {worker_id}] {data.hex()}")
                
                decoded = data.decode('utf-8', errors='ignore')
                if '}{' in decoded: decoded = decoded.replace('}{', '}\n{')
                buffer += decoded
                
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        self.process_msg(conn, worker_id, line.strip())
        except Exception as e:
            print(f"[ERR {worker_id}] {e}")
        finally:
            with self.buffer_lock:
                if worker_id in self.workers: del self.workers[worker_id]
            conn.close()
            print(f"[DISC] {worker_id}")

    def process_msg(self, conn, worker_id, line):
        try:
            msg = json.loads(line)
        except:
            print(f"[JSON-ERR {worker_id}] {line}")
            return
            
        method = msg.get('method')
        msg_id = msg.get('id')
        
        if method == 'mining.subscribe':
            # EXACT HANDSHAKE FROM s9_experiments/config.py
            response = {
                "id": msg_id,
                "result": [
                    [["mining.set_difficulty", "deadbeef"], ["mining.notify", "deadbeef"]],
                    "deadbeef", # extranonce1
                    4           # extranonce2_size
                ],
                "error": None
            }
            self._send(conn, response)
            print(f"[HANDSHAKE] Subscribe {worker_id}")
            
        elif method == 'mining.authorize':
            self._send(conn, {"id": msg_id, "result": True, "error": None})
            # Standard sequence: Set Difficulty -> Notify Job
            self._send(conn, {"id": None, "method": "mining.set_difficulty", "params": [DIFFICULTY]})
            with self.buffer_lock: self.workers[worker_id] = conn
            self.push_job(conn)
            print(f"[AUTH] {worker_id}")
            
        elif method == 'mining.submit':
            print(f"[SUBMIT] {worker_id} sending entropy!")
            params = msg.get('params', [])
            if len(params) >= 5:
                # noise: extranonce2 (params[2]) + ntime (params[3]) + nonce (params[4])
                # We also add the job_id (params[1]) for extra jitter? 
                entropy = f"{params[2]}{params[3]}{params[4]}"
                with self.buffer_lock:
                    self.entropy_buffer.append(entropy)
            
            # Response must be true
            self._send(conn, {"id": msg_id, "result": True, "error": None})
            
        elif method == 'mining.multi_version':
            self._send(conn, {"id": msg_id, "result": True, "error": None})
            
        elif not method and 'result' in msg:
            # Result of previous command (e.g. set_difficulty)
            pass
        else:
            # Catch-all
            print(f"[UNKNOWN {worker_id}] {line}")
            self._send(conn, {"id": msg_id, "result": True, "error": None})

    def push_job(self, conn, clean=True):
        self.job_id += 1
        j_id = f"{self.job_id:08x}"
        # Real-looking block header parameters
        prev = hashlib.sha256(f"prev{self.job_id}".encode()).hexdigest()
        coin1 = "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff2002dead"
        coin2 = "000000000100f2052a0100000043410496b538e853519c726a2c91e61ec11600ae1390813a627c66fb8be7947be63c52da7589379515d4e0a604f8141781e62294721166bf621e73a82cbf2342c858eeac00000000"
        
        params = [
            j_id, prev, coin1, coin2,
            [], "20000000", "1d00ffff", f"{int(time.time()):08x}", clean
        ]
        self._send(conn, {"id": None, "method": "mining.notify", "params": params})

    def _send(self, conn, data):
        try:
            conn.sendall((json.dumps(data) + "\n").encode())
        except: pass

    def run_api_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", API_PORT))
        server.listen(5)
        while self.running:
            try:
                client, _ = server.accept()
                data = client.recv(1024).decode()
                req = json.loads(data)
                if req.get('type') == 'get_entropy':
                    count = int(req.get('count', 1000))
                    with self.buffer_lock:
                        actual = min(count, len(self.entropy_buffer))
                        out = [self.entropy_buffer.popleft() for _ in range(actual)]
                    client.sendall((json.dumps({"entropy": out}) + "\n").encode())
                client.close()
            except: pass

if __name__ == "__main__":
    bridge = ArtBridge()
    bridge.start()
