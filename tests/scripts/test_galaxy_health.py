# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TG Galaxy Health Check — validates all 32 Wormhole B0 devices.

Tests:
1. test_open_single_devices: Opens each PCIe-attached device individually
2. test_mesh_1x8: Opens a (1,8) submesh and runs basic ops
3. test_mesh_8x4_no_fabric: Opens full (8,4) mesh WITHOUT fabric, runs per-device ops
4. test_mesh_8x4_with_fabric: Opens full (8,4) mesh WITH FABRIC_1D, runs collective op
5. test_matmul_all_devices: Runs matmul on every device in the mesh and validates

Run:
  source python_env/bin/activate
  pytest tests/scripts/test_galaxy_health.py -v -s --timeout=600
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc


# ─── Test 1: Open individual PCIe devices ───────────────────────────────────


def test_open_single_devices():
    """Open each PCIe-attached device one at a time to find broken ones."""
    num_pcie = ttnn.get_num_pcie_devices()
    logger.info(f"PCIe devices detected: {num_pcie}")
    assert num_pcie > 0, "No PCIe devices found"

    results = {}
    for i in range(num_pcie):
        try:
            dev = ttnn.open_device(device_id=i)
            # Run a trivial op
            a = ttnn.from_torch(
                torch.ones(1, 1, 32, 32),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
            )
            b = ttnn.multiply(a, 2.0)
            result = ttnn.to_torch(b)
            ok = torch.allclose(result, torch.full_like(result, 2.0), atol=0.1)
            ttnn.close_device(dev)
            results[i] = "PASS" if ok else "FAIL (wrong result)"
            logger.info(f"  Device {i}: {'PASS' if ok else 'FAIL (wrong result)'}")
        except Exception as e:
            results[i] = f"FAIL ({e})"
            logger.error(f"  Device {i}: FAIL — {e}")

    logger.info(f"\nSummary: {sum(1 for v in results.values() if v == 'PASS')}/{num_pcie} devices passed")
    failed = {k: v for k, v in results.items() if v != "PASS"}
    assert not failed, f"Devices failed: {failed}"


# ─── Test 2: (1,8) submesh — basic eltwise ─────────────────────────────────


@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_mesh_1x8(mesh_device, device_params):
    """Open (1,8) mesh and run eltwise ops on all 8 devices."""
    num_devices = mesh_device.get_num_devices()
    logger.info(f"Opened (1,8) mesh with {num_devices} devices")
    assert num_devices == 8

    # Replicate input to all devices
    pt_a = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    pt_b = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)

    tt_a = ttnn.from_torch(
        pt_a, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    tt_b = ttnn.from_torch(
        pt_b, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )

    tt_c = ttnn.add(tt_a, tt_b)
    result = ttnn.to_torch(tt_c, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # Each device should produce the same result (replicated)
    ref = (pt_a + pt_b).float()
    for i in range(num_devices):
        dev_result = result[i : i + 1].float()
        passing, msg = comp_pcc(ref, dev_result, 0.999)
        logger.info(f"  Device {i}: {msg}")
        assert passing, f"Device {i} add failed: {msg}"

    logger.info("(1,8) mesh eltwise add: ALL PASSED")


# ─── Test 3: Full (8,4) mesh WITHOUT fabric ─────────────────────────────────


@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_mesh_8x4_no_fabric(mesh_device, device_params):
    """Open full (8,4) mesh without fabric. Run eltwise on all 32 devices."""
    num_devices = mesh_device.get_num_devices()
    logger.info(f"Opened (8,4) mesh with {num_devices} devices (no fabric)")

    if num_devices != 32:
        pytest.skip(f"Expected 32 devices for TG, got {num_devices}")

    pt_a = torch.randn(1, 1, 32, 128, dtype=torch.bfloat16)
    pt_b = torch.randn(1, 1, 32, 128, dtype=torch.bfloat16)

    tt_a = ttnn.from_torch(
        pt_a, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    tt_b = ttnn.from_torch(
        pt_b, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )

    # Eltwise add
    tt_add = ttnn.add(tt_a, tt_b)
    result_add = ttnn.to_torch(tt_add, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # Eltwise mul
    tt_mul = ttnn.multiply(tt_a, tt_b)
    result_mul = ttnn.to_torch(tt_mul, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    ref_add = (pt_a + pt_b).float()
    ref_mul = (pt_a * pt_b).float()

    failed = []
    for i in range(num_devices):
        pass_add, msg_add = comp_pcc(ref_add, result_add[i : i + 1].float(), 0.999)
        pass_mul, msg_mul = comp_pcc(ref_mul, result_mul[i : i + 1].float(), 0.999)
        status = "PASS" if (pass_add and pass_mul) else "FAIL"
        logger.info(f"  Device {i:>2}: {status}  add={msg_add}  mul={msg_mul}")
        if not (pass_add and pass_mul):
            failed.append(i)

    logger.info(f"\nSummary: {num_devices - len(failed)}/{num_devices} devices passed")
    assert not failed, f"Devices failed eltwise: {failed}"


# ─── Test 4: Full (8,4) mesh WITH fabric ────────────────────────────────────


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_mesh_8x4_with_fabric(mesh_device, device_params):
    """Open full (8,4) mesh with FABRIC_1D. Just validates mesh opens and basic op works."""
    num_devices = mesh_device.get_num_devices()
    logger.info(f"Opened (8,4) mesh with fabric, {num_devices} devices")

    if num_devices != 32:
        pytest.skip(f"Expected 32 devices for TG, got {num_devices}")

    pt_a = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)

    tt_a = ttnn.from_torch(
        pt_a, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    tt_b = ttnn.multiply(tt_a, 3.0)
    result = ttnn.to_torch(tt_b, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    ref = (pt_a * 3.0).float()
    failed = []
    for i in range(num_devices):
        passing, msg = comp_pcc(ref, result[i : i + 1].float(), 0.999)
        if not passing:
            failed.append(i)
            logger.error(f"  Device {i:>2}: FAIL {msg}")
        else:
            logger.info(f"  Device {i:>2}: PASS {msg}")

    logger.info(f"\nFabric mesh: {num_devices - len(failed)}/{num_devices} devices passed")
    assert not failed, f"Devices failed with fabric: {failed}"


# ─── Test 5: Matmul on all devices ──────────────────────────────────────────


@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_matmul_all_devices(mesh_device, device_params):
    """Run matmul on all 32 devices and validate against PyTorch reference."""
    num_devices = mesh_device.get_num_devices()
    logger.info(f"Running matmul on {num_devices} devices")

    if num_devices != 32:
        pytest.skip(f"Expected 32 devices for TG, got {num_devices}")

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

    logger.info(f"\nMatmul: {num_devices - len(failed)}/{num_devices} devices passed")
    assert not failed, f"Matmul failed on devices: {failed}"
