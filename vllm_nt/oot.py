from ._ntops.patching import (
    _FUNCTION_PATCH_SPECS,
    _OPERATOR_SPECS,
    _nt_gelu_and_mul_forward,
    _nt_gemma_rms_norm_forward,
    _nt_layer_norm,
    _nt_mul_and_silu_forward,
    _nt_rms_norm_forward,
    _nt_silu_and_mul_forward,
    _nt_unquantized_embedding,
    _nt_unquantized_embedding_apply,
    _nt_unquantized_linear_apply,
    _reset_usage_state,
    ensure_registered,
    format_usage_summary,
    get_usage_summary,
    maybe_print_usage_summary,
)
from ._ntops.oot_support import (
    linear,
    paged_attention_decode,
    paged_attention_prefill,
    rope,
    sdpa,
    store_kv_cache,
)
from ._ntops.torch.attention import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)
from ._ntops.torch.gelu import gelu as nt_gelu
from ._ntops.torch.kv_cache import store_kvcache
from ._ntops.torch.kv_cache import get_kv_from_cache
from ._ntops.torch.rotary_emb import apply_rotary_emb
from ._ntops.torch.rms_norm import rms_norm as nt_rms_norm
from ._ntops.torch.sdpa import CausalVariant, scaled_dot_product_attention

ensure_registered()
