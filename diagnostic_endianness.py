#!/usr/bin/env python3
"""
ENDIANNESS DIAGNOSTIC TOOL
==========================
Tests multiple byte-swapping variations to match S9 hashing profile.
"""

import hashlib
import binascii

def double_sha256(data_bytes):
    return hashlib.sha256(hashlib.sha256(data_bytes).digest()).digest()

def reverse_bytes(hex_str):
    return "".join([hex_str[i:i+2] for i in range(len(hex_str)-2, -1, -2)])

def swap4(hex_str):
    """Swaps 4-byte chunks (standard Stratum)."""
    res = ""
    for i in range(0, len(hex_str), 8):
        chunk = hex_str[i:i+8]
        res += chunk[6:8] + chunk[4:6] + chunk[2:4] + chunk[0:2]
    return res

# Data from previous failed run
VERSION = "20000000"
PREV_HASH = "6b96efbdeaca56e08252d5ad6c5932020ff68a9812aeb56b56a307a73db2d16d"
MERKLE = "132d8ca90f361fc57dd79d62bad025d13e97f802cf96caf17819303fd389c288"
NTIME = "6964c254"
BITS = "1d00ffff"
NONCE = "a7b52bbb"

variations = [
    ("Raw", VERSION + PREV_HASH + MERKLE + NTIME + BITS + NONCE),
    ("SWAB All", swap4(VERSION) + swap4(PREV_HASH) + swap4(MERKLE) + swap4(NTIME) + swap4(BITS) + swap4(NONCE)),
    ("SWAB Header (S9 Standard)", swap4(VERSION + PREV_HASH + MERKLE + NTIME + BITS + NONCE)),
    ("Bitcoin Native (LE All)", reverse_bytes(VERSION) + reverse_bytes(PREV_HASH) + reverse_bytes(MERKLE) + reverse_bytes(NTIME) + reverse_bytes(BITS) + reverse_bytes(NONCE)),
]

target = 0x00000000ffff0000000000000000000000000000000000000000000000000000

for name, header in variations:
    res = double_sha256(binascii.unhexlify(header))[::-1].hex()
    val = int(res, 16)
    print(f"Variation: {name}")
    print(f" Hash: {res}")
    print(f" Pass: {val < target}\n")
