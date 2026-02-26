# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Partial RoPE Attention for MiniMax-M2.5.

MiniMax-M2.5 uses partial rotary embeddings: only the first `rotary_dim` dimensions
of Q and K are rotated, the remaining dimensions pass through unchanged.
With head_dim=128 and partial_rotary_factor=0.5, rotary_dim=64.

This subclass overrides the RoPE application sections of the parent Attention class.
QK norm is auto-detected by the parent class (lines 265-299 of attention.py).

Reference: models/tt_transformers/tt/attention.py
"""

import ttnn
from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.common import Mode


class PartialRoPEAttention(Attention):
    """
    Attention with partial rotary position embeddings for MiniMax-M2.5.

    Only the first `rotary_dim` dimensions of Q and K heads are rotated.
    The remaining `head_dim - rotary_dim` dimensions pass through unchanged.

    Uses the parent class for everything else (QKV projection, QK norm,
    KV cache, attention computation, output projection).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get rotary_dim from args (set in Phase 1.3 infrastructure fix)
        configuration = kwargs.get("configuration", args[2] if len(args) > 2 else None)
        if configuration is not None:
            self.rotary_dim = getattr(configuration, "rotary_dim", configuration.head_dim)
        else:
            self.rotary_dim = self.head_dim

    def _apply_partial_rope_decode(self, q_pre_rot, k_pre_rot, rot_mats):
        """
        Apply partial rotary embeddings for decode mode.

        Splits Q/K into rotated and passthrough portions, applies RoPE to the
        rotated portion, then concatenates.

        Args:
            q_pre_rot: Q heads before rotation [1, B, Q, D]
            k_pre_rot: K heads before rotation [1, B, K, D]
            rot_mats: Tuple of (cos, sin) rotation matrices

        Returns:
            q_rotated: Q heads after partial rotation
            k_rotated: K heads after partial rotation
        """
        if self.rotary_dim == self.head_dim:
            # Full rotation — fall back to standard path
            if self.args.use_qk_fused:
                q_pre_rot, k_pre_rot = self.to_qk_fused_memory_config(q_pre_rot, k_pre_rot)
                return ttnn.experimental.rotary_embedding_llama_fused_qk(
                    q_pre_rot, k_pre_rot, rot_mats[0], rot_mats[1], self.transformation_mats["decode"]
                )
            else:
                q = ttnn.experimental.rotary_embedding_llama(
                    q_pre_rot, rot_mats[0], rot_mats[1], self.transformation_mats["decode"], is_decode_mode=True
                )
                k = ttnn.experimental.rotary_embedding_llama(
                    k_pre_rot, rot_mats[0], rot_mats[1], self.transformation_mats["decode"], is_decode_mode=True
                )
                return q, k

        # Partial rotation: split, rotate first rotary_dim, concat with passthrough
        # Q: [1, B, num_q_heads, head_dim] -> split at rotary_dim
        q_rot_part = ttnn.slice(q_pre_rot, [0, 0, 0, 0], [*q_pre_rot.shape[:3], self.rotary_dim])
        q_pass_part = ttnn.slice(q_pre_rot, [0, 0, 0, self.rotary_dim], [*q_pre_rot.shape[:3], self.head_dim])

        k_rot_part = ttnn.slice(k_pre_rot, [0, 0, 0, 0], [*k_pre_rot.shape[:3], self.rotary_dim])
        k_pass_part = ttnn.slice(k_pre_rot, [0, 0, 0, self.rotary_dim], [*k_pre_rot.shape[:3], self.head_dim])

        ttnn.deallocate(q_pre_rot)
        ttnn.deallocate(k_pre_rot)

        # Apply RoPE to rotated portion only
        q_rotated_part = ttnn.experimental.rotary_embedding_llama(
            q_rot_part, rot_mats[0], rot_mats[1], self.transformation_mats["decode"], is_decode_mode=True
        )
        k_rotated_part = ttnn.experimental.rotary_embedding_llama(
            k_rot_part, rot_mats[0], rot_mats[1], self.transformation_mats["decode"], is_decode_mode=True
        )
        ttnn.deallocate(q_rot_part)
        ttnn.deallocate(k_rot_part)

        # Concatenate rotated + passthrough
        q_heads = ttnn.concat([q_rotated_part, q_pass_part], dim=-1)
        k_heads = ttnn.concat([k_rotated_part, k_pass_part], dim=-1)

        ttnn.deallocate(q_rotated_part)
        ttnn.deallocate(q_pass_part)
        ttnn.deallocate(k_rotated_part)
        ttnn.deallocate(k_pass_part)

        return q_heads, k_heads

    def _apply_partial_rope_prefill(self, q_pre_rot, k_pre_rot, rot_mats):
        """
        Apply partial rotary embeddings for prefill mode.

        Args:
            q_pre_rot: Q heads [1, Q, S, D]
            k_pre_rot: K heads [1, K, S, D]
            rot_mats: Tuple of (cos, sin) rotation matrices

        Returns:
            q_rotated: Q heads after partial rotation
            k_rotated: K heads after partial rotation
        """
        if self.rotary_dim == self.head_dim:
            # Full rotation — standard path
            if q_pre_rot.dtype != ttnn.bfloat16:
                q_pre_rot = ttnn.typecast(q_pre_rot, dtype=ttnn.bfloat16)
            q = ttnn.experimental.rotary_embedding_llama(
                q_pre_rot, rot_mats[0], rot_mats[1], self.transformation_mats["prefill"], is_decode_mode=False
            )
            if k_pre_rot.dtype != ttnn.bfloat16:
                k_pre_rot = ttnn.typecast(k_pre_rot, dtype=ttnn.bfloat16)
            k = ttnn.experimental.rotary_embedding_llama(
                k_pre_rot, rot_mats[0], rot_mats[1], self.transformation_mats["prefill"], is_decode_mode=False
            )
            return q, k

        # Partial rotation
        if q_pre_rot.dtype != ttnn.bfloat16:
            q_pre_rot = ttnn.typecast(q_pre_rot, dtype=ttnn.bfloat16)
        if k_pre_rot.dtype != ttnn.bfloat16:
            k_pre_rot = ttnn.typecast(k_pre_rot, dtype=ttnn.bfloat16)

        # Split at rotary_dim
        q_rot_part = ttnn.slice(q_pre_rot, [0, 0, 0, 0], [*q_pre_rot.shape[:3], self.rotary_dim])
        q_pass_part = ttnn.slice(q_pre_rot, [0, 0, 0, self.rotary_dim], [*q_pre_rot.shape[:3], self.head_dim])

        k_rot_part = ttnn.slice(k_pre_rot, [0, 0, 0, 0], [*k_pre_rot.shape[:3], self.rotary_dim])
        k_pass_part = ttnn.slice(k_pre_rot, [0, 0, 0, self.rotary_dim], [*k_pre_rot.shape[:3], self.head_dim])

        ttnn.deallocate(q_pre_rot)
        ttnn.deallocate(k_pre_rot)

        # Apply RoPE to rotated portion
        q_rotated_part = ttnn.experimental.rotary_embedding_llama(
            q_rot_part, rot_mats[0], rot_mats[1], self.transformation_mats["prefill"], is_decode_mode=False
        )
        k_rotated_part = ttnn.experimental.rotary_embedding_llama(
            k_rot_part, rot_mats[0], rot_mats[1], self.transformation_mats["prefill"], is_decode_mode=False
        )
        ttnn.deallocate(q_rot_part)
        ttnn.deallocate(k_rot_part)

        # Concatenate
        q_heads = ttnn.concat([q_rotated_part, q_pass_part], dim=-1)
        k_heads = ttnn.concat([k_rotated_part, k_pass_part], dim=-1)

        ttnn.deallocate(q_rotated_part)
        ttnn.deallocate(q_pass_part)
        ttnn.deallocate(k_rotated_part)
        ttnn.deallocate(k_pass_part)

        return q_heads, k_heads

    def forward_decode(self, x, current_pos, rot_mats=None, page_table=None, kv_cache=None):
        """
        Override decode forward to use partial RoPE.

        This method replaces the RoPE application section of the parent's forward_decode
        while keeping all other logic (QKV projection, QK norm, KV cache, attention, output proj).
        """
        # QKV projection (reuse parent's projection logic)
        xqkv_fused = self._compute_qkv(x)

        # Split into Q, K, V heads
        q_heads_pre_rot, k_heads_pre_rot, v_heads = self._split_qkv_decode(xqkv_fused)

        # QK norm (auto-detected by parent)
        norm_config = self.args.get_norm_config("attn", Mode.DECODE, None)
        q_heads_pre_rot = self.q_norm(q_heads_pre_rot, mode=Mode.DECODE, norm_config=norm_config)
        k_heads_pre_rot = self.k_norm(k_heads_pre_rot, mode=Mode.DECODE, norm_config=norm_config)
        ttnn.deallocate(xqkv_fused)

        # Partial RoPE
        q_heads, k_heads = self._apply_partial_rope_decode(q_heads_pre_rot, k_heads_pre_rot, rot_mats)

        # KV cache update + attention + output projection (delegate to parent)
        return self._post_rope_decode(q_heads, k_heads, v_heads, current_pos, page_table, kv_cache)

    def forward_prefill(
        self, x, rot_mats, user_id=0, page_table=None, chunk_page_table=None, chunk_start_idx=None, kv_cache=None
    ):
        """
        Override prefill forward to use partial RoPE.

        This method replaces the RoPE application section of the parent's forward_prefill
        while keeping all other logic.
        """
        # QKV projection
        xqkv_fused = self._compute_qkv_prefill(x)

        # Split into Q, K, V heads
        q_heads_pre_rot, k_heads_pre_rot, v_heads = self._split_qkv_prefill(xqkv_fused)

        # QK norm
        norm_config = self.args.get_norm_config("attn", Mode.PREFILL, None)
        q_heads_pre_rot = self.q_norm(q_heads_pre_rot, mode=Mode.PREFILL, norm_config=norm_config)
        k_heads_pre_rot = self.k_norm(k_heads_pre_rot, mode=Mode.PREFILL, norm_config=norm_config)
        ttnn.deallocate(xqkv_fused)

        # Partial RoPE
        q_heads, k_heads = self._apply_partial_rope_prefill(q_heads_pre_rot, k_heads_pre_rot, rot_mats)

        # KV cache fill + attention + output projection
        return self._post_rope_prefill(
            q_heads, k_heads, v_heads, user_id, page_table, chunk_page_table, chunk_start_idx, kv_cache
        )
