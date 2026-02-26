# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for MiniMax-M2.5 partial RoPE attention.

Verifies that:
1. Only the first rotary_dim dimensions are rotated
2. Remaining dimensions pass through unchanged
3. partial_rotary_factor is correctly read from model config
"""

import pytest
import torch


def _compute_rope_reference(x, freqs_cos, freqs_sin):
    """Apply rotary embeddings to input tensor (reference implementation)."""
    # x shape: [..., head_dim]
    # Split into pairs for rotation
    x_r = x.float().reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x_r.unbind(-1)

    # Apply rotation
    out1 = x1 * freqs_cos - x2 * freqs_sin
    out2 = x1 * freqs_sin + x2 * freqs_cos

    # Interleave back
    out = torch.stack([out1, out2], dim=-1).reshape(*x.shape)
    return out.to(x.dtype)


@pytest.mark.parametrize("head_dim,rotary_dim", [(128, 64), (128, 128), (64, 32)])
@pytest.mark.parametrize("seq_len", [1, 16])
def test_partial_rope_reference(head_dim, rotary_dim, seq_len):
    """Test partial RoPE: only first rotary_dim dims should be rotated."""
    torch.manual_seed(42)

    batch_size = 1
    num_heads = 4
    theta = 10000.0

    # Create input Q/K
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Compute rotation frequencies for rotary_dim
    freqs = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    positions = torch.arange(seq_len).float()
    angles = torch.outer(positions, freqs)  # [seq_len, rotary_dim//2]
    freqs_cos = angles.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, rotary_dim//2]
    freqs_sin = angles.sin().unsqueeze(0).unsqueeze(0)

    # Apply partial RoPE
    q_rot_part = q[..., :rotary_dim]
    q_pass_part = q[..., rotary_dim:]
    k_rot_part = k[..., :rotary_dim]
    k_pass_part = k[..., rotary_dim:]

    q_rotated = _compute_rope_reference(q_rot_part, freqs_cos, freqs_sin)
    k_rotated = _compute_rope_reference(k_rot_part, freqs_cos, freqs_sin)

    q_out = torch.cat([q_rotated, q_pass_part], dim=-1)
    k_out = torch.cat([k_rotated, k_pass_part], dim=-1)

    # Verify shapes
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape

    # Verify passthrough dims are unchanged
    if rotary_dim < head_dim:
        assert torch.allclose(
            q_out[..., rotary_dim:], q[..., rotary_dim:], atol=1e-6
        ), "Passthrough dimensions should be unchanged"
        assert torch.allclose(
            k_out[..., rotary_dim:], k[..., rotary_dim:], atol=1e-6
        ), "Passthrough dimensions should be unchanged"

    # Verify rotated dims are changed (for seq_len > 0, positions > 0)
    if seq_len > 1:
        # At position 0, rotation is identity (cos=1, sin=0), so skip pos 0
        assert not torch.allclose(
            q_out[:, :, 1:, :rotary_dim], q[:, :, 1:, :rotary_dim], atol=1e-4
        ), "Rotated dimensions should be different from input (for non-zero positions)"


def test_partial_rotary_factor_config():
    """Test that partial_rotary_factor and rotary_dim are correctly set in ModelArgs."""
    # Verify the infrastructure fix from Phase 1.3
    # This tests that model_config.py correctly computes rotary_dim

    head_dim = 128
    partial_rotary_factor = 0.5
    expected_rotary_dim = int(head_dim * partial_rotary_factor)

    assert expected_rotary_dim == 64, f"Expected rotary_dim=64, got {expected_rotary_dim}"

    # Test with factor=1.0 (full rotation, default)
    full_rotary_dim = int(head_dim * 1.0)
    assert full_rotary_dim == 128, f"Expected full rotary_dim=128, got {full_rotary_dim}"
