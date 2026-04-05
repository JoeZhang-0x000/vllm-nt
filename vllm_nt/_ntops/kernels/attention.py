import functools
import math

import ninetoothed
import ninetoothed.language as ntl
import torch
from ninetoothed import Tensor


class _VarLen:
    BLOCK_SIZE_M = ninetoothed.block_size()
    BLOCK_SIZE_N = ninetoothed.block_size()

    @staticmethod
    def arrangement(
        q, k, v, o, sm_scale, is_causal, block_size_m=None, block_size_n=None
    ):
        if block_size_m is None:
            block_size_m = _VarLen.BLOCK_SIZE_M
        if block_size_n is None:
            block_size_n = _VarLen.BLOCK_SIZE_N

        def _arrange_qo(x):
            arranged = (
                x.permute((0, 2, 3, 1, 4)).tile((1, 1, 1, block_size_m, -1))
            )
            arranged.dtype = arranged.dtype.squeeze((0, 1, 2))
            return arranged

        def _arrange_kv(x):
            arranged = (
                x.permute((0, 2, 3, 1, 4))
                .tile((1, 1, 1, block_size_n, -1))
                .expand((-1, -1, q_arranged.shape[2], q_arranged.shape[3], -1))
                .tile((1, 1, 1, -1, -1))
            )
            arranged.dtype = arranged.dtype.squeeze((0, 1, 2, 4))
            arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1, 2))
            return arranged

        q_arranged = _arrange_qo(q)
        o_arranged = _arrange_qo(o)
        k_arranged = _arrange_kv(k)
        v_arranged = _arrange_kv(v)
        return q_arranged, k_arranged, v_arranged, o_arranged, sm_scale, is_causal

    @staticmethod
    def application(q, k, v, o, sm_scale, is_causal):
        q_i = ntl.cast(q, dtype=ntl.float32) * sm_scale
        m_i = ntl.full((q.shape[0],), float("-inf"), dtype=ntl.float32)
        l_i = ntl.zeros((q.shape[0],), dtype=ntl.float32)
        o_i = ntl.zeros(o.shape, dtype=ntl.float32)

        for j in range(k.shape[0]):
            k_j = ntl.cast(k[j], dtype=ntl.float32)
            v_j = ntl.cast(v[j], dtype=ntl.float32)
            s_ij = ntl.dot(q_i, ntl.trans(k_j))
            s_ij = ntl.where(
                k[j].offsets(1)[None, :] < k.source.shape[1], s_ij, float("-inf")
            )
            if is_causal:
                causal_mask = q.offsets(1)[:, None] >= k[j].offsets(1)[None, :]
                s_ij = ntl.where(causal_mask, s_ij, float("-inf"))
            m_ij = ntl.max(s_ij, axis=1)
            m_i_new = ntl.maximum(m_ij, m_i)
            p_ij = ntl.exp(s_ij - m_i_new[:, None])
            l_ij = ntl.sum(p_ij, axis=1)
            exp_diff = ntl.exp(m_i - m_i_new)
            l_i_new = l_i * exp_diff + l_ij
            o_i = (
                o_i * (l_i / l_i_new * exp_diff)[:, None]
                + ntl.dot(p_ij, v_j) / l_i_new[:, None]
            )
            m_i = m_i_new
            l_i = l_i_new
        o = ntl.cast(o_i, dtype=o.dtype)  # noqa: F841


def varlen_premake(block_size_m=None, block_size_n=None):
    arrangement_ = functools.partial(
        _VarLen.arrangement,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
    )
    shape_options = (None, None, None, None, {"constexpr": True, "upper_bound": 128})
    tensors = (
        Tensor(5, jagged_dim=1, shape_options=shape_options),
        Tensor(5, jagged_dim=1, shape_options=shape_options),
        Tensor(5, jagged_dim=1, shape_options=shape_options),
        Tensor(5, jagged_dim=1, shape_options=shape_options),
        Tensor(0),
        Tensor(0),
    )
    return arrangement_, _VarLen.application, tensors


class KvCache:
    @staticmethod
    def arrangement(
        q, k_cache, v_cache, o, cache_seqlens, block_table, sm_scale, is_causal
    ):
        def _arrange_qo(x):
            arranged = x.tile((1, 1, 1, -1))
            arranged.dtype = arranged.dtype.squeeze((0, 1))
            return arranged

        q_arranged = _arrange_qo(q)
        o_arranged = _arrange_qo(o)

        def _arrange_cache(x):
            arranged = (
                x.permute((0, 2, 1, 3))
                .tile((1, 1, -1, -1))
                .tile((-1, 1, 1, 1))
                .permute((0, 2, 1, 3))
                .expand((q_arranged.shape[0], -1, -1, -1))
            )
            arranged.dtype = arranged.dtype.squeeze((1, 2, 3))
            arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))
            return arranged

        k_cache_arranged = _arrange_cache(k_cache)
        v_cache_arranged = _arrange_cache(v_cache)

        cache_seqlens_arranged = cache_seqlens.tile((1, 1, 1, 1)).expand(
            (-1, q_arranged.shape[1], q_arranged.shape[2], q_arranged.shape[3])
        )
        cache_seqlens_arranged.dtype = cache_seqlens_arranged.dtype.squeeze((1, 2, 3))

        block_table_arranged = block_table.tile((1, -1, 1, 1)).expand(
            (-1, q_arranged.shape[1], q_arranged.shape[2], q_arranged.shape[3])
        )
        block_table_arranged.dtype = block_table_arranged.dtype.squeeze((0, 2, 3))

        return (
            q_arranged,
            k_cache_arranged,
            v_cache_arranged,
            o_arranged,
            cache_seqlens_arranged,
            block_table_arranged,
            sm_scale,
            is_causal,
        )

    @staticmethod
    def application(
        q, k_cache, v_cache, o, cache_seqlens, block_table, sm_scale, is_causal
    ):
        q_i = ntl.cast(q, dtype=ntl.float32) * sm_scale
        m_i = ntl.full((1,), float("-inf"), dtype=ntl.float32)
        l_i = ntl.full((1,), float(0), dtype=ntl.float32)
        o_i = ntl.zeros(q.shape, dtype=ntl.float32)

        block_nums = block_table.shape[0]
        block_size = k_cache[0].shape[0]
        seq_start = 0
        for blk in range(block_nums):
            blk_id = block_table[blk]
            k_j = ntl.cast(k_cache[blk_id], dtype=ntl.float32)
            v_j = ntl.cast(v_cache[blk_id], dtype=ntl.float32)
            s_ij = ntl.dot(q_i, ntl.trans(k_j))

            mask = (k_cache[blk].offsets(1) % block_size + seq_start) < cache_seqlens[0]
            s_ij = ntl.where(mask[None, :], s_ij, float("-inf"))

            if is_causal:
                pass

            m_ij = ntl.max(s_ij, axis=1)
            m_i_new = ntl.maximum(m_ij, m_i)
            p_ij = ntl.exp(s_ij - m_i_new[:, None])
            l_ij = ntl.sum(p_ij, axis=1)
            exp_diff = ntl.exp(m_i - m_i_new)
            l_i_new = l_i * exp_diff + l_ij
            o_i = (
                o_i * (l_i / l_i_new * exp_diff)[:, None]
                + ntl.dot(p_ij, v_j) / l_i_new[:, None]
            )
            m_i = m_i_new
            l_i = l_i_new
            seq_start += block_size
        o = ntl.cast(o_i, o.dtype)  # noqa: F841


def cache_premake():
    tensors = (
        Tensor(4, shape_options=(None, None, None, {"constexpr": True})),
        Tensor(4, shape_options={"constexpr": True}),
        Tensor(4, shape_options={"constexpr": True}),
        Tensor(4, shape_options=(None, None, None, {"constexpr": True})),
        Tensor(4),
        Tensor(4),
        Tensor(0),
        Tensor(0),
    )
    return KvCache.arrangement, KvCache.application, tensors


def reference_scale(query, sm_scale=None):
    return 1 / math.sqrt(query.shape[-1]) if sm_scale is None else sm_scale
