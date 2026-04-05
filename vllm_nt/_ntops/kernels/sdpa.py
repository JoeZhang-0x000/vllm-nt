import enum
import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

BLOCK_SIZE_M = ninetoothed.block_size()
BLOCK_SIZE_N = ninetoothed.block_size()


class CausalVariant(enum.IntEnum):
    UPPER_LEFT = enum.auto()
    LOWER_RIGHT = enum.auto()


def arrangement(
    query,
    key,
    value,
    present_key,
    present_value,
    present_key_slot,
    present_value_slot,
    attn_mask,
    is_causal,
    scale,
    output,
    with_attn_mask,
    causal_variant,
    with_kv_cache,
    block_size_m=None,
    block_size_n=None,
):
    def _arrange_query_or_output(input_):
        arranged = input_.tile((1, 1, block_size_m, -1)).tile(
            (1, query.shape[-3] // key.shape[-3], 1, 1)
        )
        arranged.dtype = arranged.dtype.squeeze((0, 2, 3))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))
        return arranged

    def _arrange_key_or_value(input_):
        arranged = (
            input_.tile((1, 1, block_size_n, -1))
            .tile((1, 1, -1, -1))
            .expand((-1, -1, query_arranged.shape[-2], -1))
        )
        arranged.dtype = arranged.dtype.squeeze((0, 1, 3))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))
        return arranged

    def _arrange_present(input_):
        arranged = input_.tile((1, 1, block_size_m, block_size_n))
        arranged.dtype = arranged.dtype.squeeze((0, 1))
        return arranged

    def _arrange_attn_mask(input_):
        arranged = input_.tile((1, 1, block_size_m, block_size_n)).tile((1, 1, 1, -1))
        arranged.dtype = arranged.dtype.squeeze((0, 1, 2))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))
        return arranged

    if block_size_m is None:
        block_size_m = BLOCK_SIZE_M
    if block_size_n is None:
        block_size_n = BLOCK_SIZE_N

    query_arranged = _arrange_query_or_output(query)
    key_arranged = _arrange_key_or_value(key)
    value_arranged = _arrange_key_or_value(value)
    present_key_arranged = _arrange_present(present_key)
    present_value_arranged = _arrange_present(present_value)
    present_key_slot_arranged = _arrange_present(present_key_slot)
    present_value_slot_arranged = _arrange_present(present_value_slot)
    attn_mask_arranged = _arrange_attn_mask(attn_mask)
    output_arranged = _arrange_query_or_output(output)

    if with_kv_cache:
        return (
            query_arranged,
            key_arranged,
            value_arranged,
            present_key_arranged,
            present_value_arranged,
            present_key_slot_arranged,
            present_value_slot_arranged,
            attn_mask_arranged,
            is_causal,
            scale,
            output_arranged,
            with_attn_mask,
            causal_variant,
        )

    return (
        query_arranged,
        key_arranged,
        value_arranged,
        attn_mask_arranged,
        is_causal,
        scale,
        output_arranged,
        with_attn_mask,
        causal_variant,
    )


def application_with_kv_cache(
    query,
    key,
    value,
    present_key,
    present_value,
    present_key_slot,
    present_value_slot,
    attn_mask,
    is_causal,
    scale,
    output,
    with_attn_mask,
    causal_variant,
):
    present_key_slot = present_key  # noqa: F841
    present_value_slot = present_value  # noqa: F841
    application_without_kv_cache(
        query,
        key,
        value,
        attn_mask,
        is_causal,
        scale,
        output,
        with_attn_mask,
        causal_variant,
    )


def application_without_kv_cache(
    query,
    key,
    value,
    attn_mask,
    is_causal,
    scale,
    output,
    with_attn_mask,
    causal_variant,
):
    for i in range(query.shape[0]):
        query_i = (1.4426950408889634 * scale * query[i]).to(query[i].dtype)
        acc = ntl.zeros((query_i.shape[-2], query_i.shape[-1]), dtype=ntl.float32)
        lse = ntl.full((query_i.shape[-2],), 1, dtype=ntl.float32)
        max_ = ntl.full((query_i.shape[-2],), float("-inf"), dtype=ntl.float32)

        for j in range(key.shape[0]):
            qk = ntl.dot(query_i, ntl.trans(key[j]))
            qk = ntl.where(key[j].offsets(-2) < key.source.shape[-2], qk, float("-inf"))

            if with_attn_mask:
                qk += attn_mask[j]

            if is_causal:
                if causal_variant == int(CausalVariant.LOWER_RIGHT):
                    mask = (
                        query[i].offsets(-2)[:, None]
                        + key.source.shape[-2]
                        - query.source.shape[-2]
                        >= key[j].offsets(-2)[None, :]
                    )
                else:
                    mask = query[i].offsets(-2)[:, None] >= key[j].offsets(-2)[None, :]
                qk = ntl.where(mask, qk, float("-inf"))

            next_max = ntl.maximum(max_, ntl.max(qk, 1))
            stable_qk = ntl.exp2(qk - next_max[:, None])
            alpha = ntl.exp2(max_ - next_max)
            acc = acc * alpha[:, None] + ntl.dot(
                stable_qk.to(value[i].dtype), value[j]
            )
            max_ = next_max
            lse = lse * alpha + ntl.sum(stable_qk, 1)

        acc /= lse[:, None]
        output[i] = acc  # noqa: F841


def premake(
    with_kv_cache,
    emb_dim=None,
    is_causal=None,
    with_attn_mask=None,
    causal_variant=None,
    dtype=None,
    block_size_m=None,
    block_size_n=None,
):
    arrangement_ = functools.partial(
        arrangement,
        with_kv_cache=with_kv_cache,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
    )

    query, key, value, attn_mask, output = (
        Tensor(
            4,
            dtype=dtype,
            shape_options=(None, None, None, {"constexpr": True, "upper_bound": 128}),
        )
        for _ in range(5)
    )
    present_key, present_value, present_key_slot, present_value_slot = (
        Tensor(4, dtype=dtype) for _ in range(4)
    )
    scale = Tensor(0, dtype=ninetoothed.float64)
    is_causal_t = Tensor(0, constexpr=True, value=is_causal)
    with_attn_mask_t = Tensor(0, constexpr=True, value=with_attn_mask)
    causal_variant_t = Tensor(0, constexpr=True, value=causal_variant)

    if emb_dim is not None:
        for tensor in (query, key, value, attn_mask, output):
            tensor.shape = tensor.shape[:-1] + (emb_dim,)

    application = (
        application_with_kv_cache if with_kv_cache else application_without_kv_cache
    )
    tensors = (
        query,
        key,
        value,
        present_key,
        present_value,
        present_key_slot,
        present_value_slot,
        attn_mask,
        is_causal_t,
        scale,
        output,
        with_attn_mask_t,
        causal_variant_t,
    )
    return arrangement_, application, tensors
