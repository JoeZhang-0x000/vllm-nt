import pytest

torch = pytest.importorskip("torch")


def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_mlu  # noqa: F401

        if torch.mlu.is_available():
            return "mlu"
    except ImportError:
        pass
    try:
        import torch_musa  # noqa: F401

        if torch.musa.is_available():
            return "musa"
    except ImportError:
        pass
    pytest.skip("No accelerator device available (need CUDA, MLU, or MUSA)")


class TestNTKVCache:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_store_kvcache_writes_selected_slots(self, dtype):
        device = _get_device()
        from vllm_nt._ntops.torch.kv_cache import store_kvcache

        key = torch.randn((3, 2, 4), dtype=dtype, device=device)
        value = torch.randn((3, 2, 4), dtype=dtype, device=device)
        k_cache = torch.zeros((8, 8), dtype=dtype, device=device)
        v_cache = torch.zeros((8, 8), dtype=dtype, device=device)
        slot_mapping = torch.tensor([1, 4, 6], dtype=torch.int32, device=device)

        store_kvcache(key, value, k_cache, v_cache, slot_mapping)

        torch.testing.assert_close(k_cache[1], key[0].reshape(-1))
        torch.testing.assert_close(k_cache[4], key[1].reshape(-1))
        torch.testing.assert_close(k_cache[6], key[2].reshape(-1))
        torch.testing.assert_close(v_cache[1], value[0].reshape(-1))
        torch.testing.assert_close(v_cache[4], value[1].reshape(-1))
        torch.testing.assert_close(v_cache[6], value[2].reshape(-1))
