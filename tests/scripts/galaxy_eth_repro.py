#!/usr/bin/env python3
"""
Minimal reproducer: TG Galaxy ETH training hang.

Demonstrates that ttnn.open_mesh_device() hangs indefinitely due to
broken ETH core training on eth cores (1,6) and (8,6).

Hardware is healthy (pyluwen detects and inits all 32 chips instantly),
but the ttnn device driver cannot open because topology discovery
busy-waits on ETH training that never completes.

Host: giganoto (TG Galaxy, 32x Wormhole B0)
Firmware: 19.4.2.0
Driver: TT-KMD 2.6.1-pre

Usage:
  source python_env/bin/activate
  python3 tests/scripts/galaxy_eth_repro.py
"""

import time

# ── Step 1: Prove hardware is alive via pyluwen (no ETH training) ──
from pyluwen import detect_chips_fallible

t0 = time.time()
chips = detect_chips_fallible()
print(f"[PASS] pyluwen detected {len(chips)} chips in {time.time()-t0:.2f}s (all OK, no None)")
assert all(c is not None for c in chips), "Some chips not detected!"

for i in range(min(4, len(chips))):
    pci = chips[i].init()
    print(f"  Chip {i}: init OK")

# ── Step 2: Show ttnn.open_mesh_device hangs ──
import ttnn

print(f"\nAttempting ttnn.open_mesh_device(MeshShape(1,1)) ...")
print(f"  This will hang for ~30 min (2x 15-min ETH training timeouts on cores 1,6 and 8,6)")
print(f"  Then hang indefinitely at 'Opening user mode device driver'")
print(f"  Kill with Ctrl-C when satisfied.\n")

t0 = time.time()
try:
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    elapsed = time.time() - t0
    n = mesh.get_num_devices()
    print(f"[PASS] Mesh opened in {elapsed:.1f}s with {n} device(s)")
    ttnn.close_mesh_device(mesh)
except KeyboardInterrupt:
    print(f"\n[HANG] Killed after {time.time()-t0:.0f}s — ttnn.open_mesh_device never returned")
except Exception as e:
    print(f"[FAIL] {e}")
