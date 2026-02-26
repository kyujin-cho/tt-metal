#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TG Galaxy Health Check — standalone script (no pytest fixtures).

Validates all 32 Wormhole B0 devices by:
1. Discovering topology with no_wait_for_eth_training to skip the 15-min hang
2. Opening a mesh device
3. Running eltwise (add, mul) and matmul on every device
4. Reporting per-device pass/fail

Usage:
  source python_env/bin/activate
  TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 python3 tests/scripts/galaxy_health_check.py          # TG Galaxy (8x4)
  TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 python3 tests/scripts/galaxy_health_check.py 1x8      # T3K (1x8)
  TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 python3 tests/scripts/galaxy_health_check.py 1 4      # 4-device
"""

import os
import sys
import time
import signal

import torch
from loguru import logger

# Set env before importing ttnn
os.environ.setdefault("TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN", "1")

import ttnn
from models.common.utility_functions import comp_pcc


def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")


def test_eltwise(mesh_device, num_devices):
    """Run eltwise add and multiply on all devices."""
    logger.info("=== Eltwise Test (add + multiply) ===")

    pt_a = torch.randn(1, 1, 32, 128, dtype=torch.bfloat16)
    pt_b = torch.randn(1, 1, 32, 128, dtype=torch.bfloat16)

    tt_a = ttnn.from_torch(
        pt_a, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    tt_b = ttnn.from_torch(
        pt_b, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )

    tt_add = ttnn.add(tt_a, tt_b)
    tt_mul = ttnn.multiply(tt_a, tt_b)

    result_add = ttnn.to_torch(tt_add, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    result_mul = ttnn.to_torch(tt_mul, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    ref_add = (pt_a + pt_b).float()
    ref_mul = (pt_a * pt_b).float()

    failed = []
    for i in range(num_devices):
        pass_add, msg_add = comp_pcc(ref_add, result_add[i : i + 1].float(), 0.999)
        pass_mul, msg_mul = comp_pcc(ref_mul, result_mul[i : i + 1].float(), 0.999)
        status = "PASS" if (pass_add and pass_mul) else "FAIL"
        logger.info(f"  Device {i:>2}: {status}  add_pcc={msg_add}  mul_pcc={msg_mul}")
        if not (pass_add and pass_mul):
            failed.append(i)

    ttnn.deallocate(tt_a)
    ttnn.deallocate(tt_b)
    ttnn.deallocate(tt_add)
    ttnn.deallocate(tt_mul)

    return failed


def test_matmul(mesh_device, num_devices):
    """Run matmul on all devices."""
    logger.info("=== Matmul Test (32x128 @ 128x64) ===")

    M, K, N = 32, 128, 64
    pt_a = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    pt_b = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    ref = pt_a.float() @ pt_b.float()

    tt_a = ttnn.from_torch(
        pt_a, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    tt_b = ttnn.from_torch(
        pt_b, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )

    tt_c = ttnn.matmul(tt_a, tt_b)
    result = ttnn.to_torch(tt_c, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    failed = []
    for i in range(num_devices):
        passing, msg = comp_pcc(ref, result[i : i + 1].float(), 0.98)
        status = "PASS" if passing else "FAIL"
        logger.info(f"  Device {i:>2}: {status} {msg}")
        if not passing:
            failed.append(i)

    ttnn.deallocate(tt_a)
    ttnn.deallocate(tt_b)
    ttnn.deallocate(tt_c)

    return failed


def test_larger_matmul(mesh_device, num_devices):
    """Run a larger matmul to stress-test devices."""
    logger.info("=== Large Matmul Test (256x1024 @ 1024x512) ===")

    M, K, N = 256, 1024, 512
    pt_a = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    pt_b = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    ref = pt_a.float() @ pt_b.float()

    tt_a = ttnn.from_torch(
        pt_a, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    tt_b = ttnn.from_torch(
        pt_b, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )

    tt_c = ttnn.matmul(tt_a, tt_b)
    result = ttnn.to_torch(tt_c, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    failed = []
    for i in range(num_devices):
        passing, msg = comp_pcc(ref, result[i : i + 1].float(), 0.97)
        status = "PASS" if passing else "FAIL"
        logger.info(f"  Device {i:>2}: {status} {msg}")
        if not passing:
            failed.append(i)

    ttnn.deallocate(tt_a)
    ttnn.deallocate(tt_b)
    ttnn.deallocate(tt_c)

    return failed


def main():
    torch.manual_seed(42)

    # Parse mesh shape from args or default to TG Galaxy (8x4)
    if len(sys.argv) >= 3:
        rows, cols = int(sys.argv[1]), int(sys.argv[2])
    elif len(sys.argv) == 2 and "x" in sys.argv[1]:
        rows, cols = [int(x) for x in sys.argv[1].split("x")]
    else:
        # Default: TG Galaxy = 8x4 = 32 devices
        rows, cols = 8, 4

    mesh_shape = ttnn.MeshShape(rows, cols)
    num_devices = rows * cols
    label = f"Mesh ({rows}x{cols}) — {num_devices} devices"
    logger.info(f"Target: {label}")

    # ── Open mesh ──
    # NOTE: On Galaxy (UBB), ETH training may take up to 15 min per stuck core.
    # The UBB handler logs a warning and continues (does NOT throw).
    # We must wait it out — do NOT set a short timeout.
    logger.info(f"Opening mesh device {rows}x{cols} (ETH training may take up to 15 min)...")
    t0 = time.time()

    try:
        mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)
    except Exception as e:
        logger.error(f"Failed to open mesh: {e}")
        sys.exit(3)

    elapsed = time.time() - t0
    actual_devices = mesh_device.get_num_devices()
    logger.info(f"Mesh opened in {elapsed:.1f}s with {actual_devices} devices")

    # ── Step 4: Run tests ──
    all_failed = {}
    try:
        failed = test_eltwise(mesh_device, actual_devices)
        if failed:
            all_failed["eltwise"] = failed

        failed = test_matmul(mesh_device, actual_devices)
        if failed:
            all_failed["matmul"] = failed

        failed = test_larger_matmul(mesh_device, actual_devices)
        if failed:
            all_failed["large_matmul"] = failed

    finally:
        # ── Step 5: Cleanup ──
        logger.info("Closing mesh device...")
        ttnn.close_mesh_device(mesh_device)

    # ── Step 6: Report ──
    logger.info("\n" + "=" * 60)
    logger.info(f"GALAXY HEALTH CHECK RESULTS — {label}")
    logger.info("=" * 60)

    if not all_failed:
        logger.info(f"ALL {actual_devices} DEVICES PASSED all tests")
        logger.info("  Eltwise (add, mul): PASS")
        logger.info("  Matmul (32x128x64): PASS")
        logger.info("  Large Matmul (256x1024x512): PASS")
        sys.exit(0)
    else:
        for test_name, devices in all_failed.items():
            logger.error(f"  {test_name}: FAILED on devices {devices}")
        sys.exit(1)


if __name__ == "__main__":
    main()
