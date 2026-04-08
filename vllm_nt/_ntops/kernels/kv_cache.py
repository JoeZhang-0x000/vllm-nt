from ninetoothed import Tensor


def arrangement(key, value, k_cache, v_cache, slot_mapping):
    num_tokens = key.shape[0]
    hidden_size = key.shape[1]

    def _arrange_kv(x):
        arranged = x.tile((1, hidden_size)).squeeze((1,))
        arranged.dtype = arranged.dtype.squeeze((0,))
        return arranged

    def _arrange_cache(x):
        arranged = x.tile((1, -1)).squeeze((1,)).tile((-1,)).expand((num_tokens,))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0,))
        return arranged

    key_arranged = _arrange_kv(key)
    value_arranged = _arrange_kv(value)
    k_cache_arranged = _arrange_cache(k_cache)
    v_cache_arranged = _arrange_cache(v_cache)
    slot_mapping_arranged = slot_mapping.tile((1,))
    return (
        key_arranged,
        value_arranged,
        k_cache_arranged,
        v_cache_arranged,
        slot_mapping_arranged,
    )


def application(key, value, k_cache, v_cache, slot_mapping):
    k_cache[slot_mapping] = key
    v_cache[slot_mapping] = value


def premake():
    tensors = (
        Tensor(2, shape_options=({"constexpr": True}, {"constexpr": True})),
        Tensor(2, shape_options=({"constexpr": True}, {"constexpr": True})),
        Tensor(2, shape_options=({"constexpr": True}, {"constexpr": True})),
        Tensor(2, shape_options=({"constexpr": True}, {"constexpr": True})),
        Tensor(1, shape_options=({"constexpr": True},)),
    )
    return arrangement, application, tensors
