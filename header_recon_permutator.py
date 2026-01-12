#!/usr/bin/env python3
"""
EXHAUSTIVE HEADER RECONSTRUCTION (S9 BM1387)
============================================
Tests all logical byte-order combinations to find the exact header hashed by the ASIC.
"""

import hashlib
import binascii
import itertools

def double_sha256(data_bytes):
    return hashlib.sha256(hashlib.sha256(data_bytes).digest()).digest()

def swab32(hex_str):
    res = ""
    for i in range(0, len(hex_str), 8):
        chunk = hex_str[i:i+8]
        res += chunk[6:8] + chunk[4:6] + chunk[2:4] + chunk[0:2]
    return res

def reverse_bytes(hex_str):
    return "".join([hex_str[i:i+2] for i in range(len(hex_str)-2, -1, -2)])

# Captured Data from previous run
V_RAW = "20000000"
P_RAW = "6b96efbdeaca56e08252d5ad6c5932020ff68a9812aeb56b56a307a73db2d16d"
EN2 = "04020000"
NT_RAW = "6964c254"
NB_RAW = "1d00ffff"
NO_RAW = "a7b52bbb"

# Merkle calculation dependency
EX1 = "00000000"
CB1 = "00" * 32
CB2 = "00" * 32

def test_everything():
    target = 0x00000000ffff0000000000000000000000000000000000000000000000000000
    
    # Define candidates for each field
    candidates = {
        'v': [V_RAW, reverse_bytes(V_RAW), swab32(V_RAW)],
        'p': [P_RAW, reverse_bytes(P_RAW), swab32(P_RAW)],
        'nt': [NT_RAW, reverse_bytes(NT_RAW), swab32(NT_RAW)],
        'nb': [NB_RAW, reverse_bytes(NB_RAW), swab32(NB_RAW)],
        'no': [NO_RAW, reverse_bytes(NO_RAW), swab32(NO_RAW)],
        'ex2': [EN2, reverse_bytes(EN2), swab32(EN2)],
    }
    
    print("[RECON] Searching for valid PoW combinations...")
    
    count = 0
    # Iterate through all combinations of endianness for each field
    for v, p, nt, nb, no, ex2 in itertools.product(
        candidates['v'], candidates['p'], candidates['nt'], 
        candidates['nb'], candidates['no'], candidates['ex2']
    ):
        # Merkle Root must be recalculated for each ex2 variety
        cb = binascii.unhexlify(CB1 + EX1 + ex2 + CB2)
        mr_be = double_sha256(cb).hex()
        
        # Test Merkle Varieties
        for mr in [mr_be, reverse_bytes(mr_be), swab32(mr_be)]:
            header = v + p + mr + nt + nb + no
            try:
                h_bytes = binascii.unhexlify(header)
                res_hash = double_sha256(h_bytes)[::-1].hex()
                val = int(res_hash, 16)
                
                if val < target:
                    print(f"\n[MATCH FOUND!]")
                    print(f" Header: {header}")
                    print(f" Hash:   {res_hash}")
                    print(f" Models: V:{v}, P:{p}, MR:{mr}, T:{nt}, B:{nb}, N:{no}")
                    return True
            except: pass
            count += 1
            if count % 1000 == 0:
                print(f" Progress: {count} tested...", end='\r')
                
    print(f"\n[FAIL] Searched {count} combinations. No match found.")
    return False

if __name__ == "__main__":
    test_everything()
