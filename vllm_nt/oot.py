from ._ntops.patching import (
    _OPERATOR_SPECS,
    _nt_gelu_and_mul_forward,
    _nt_gemma_rms_norm_forward,
    _nt_mul_and_silu_forward,
    _nt_rms_norm_forward,
    _nt_silu_and_mul_forward,
    _nt_unquantized_embedding,
    _nt_unquantized_linear_apply,
    _reset_usage_state,
    ensure_registered,
    format_usage_summary,
    get_usage_summary,
    maybe_print_usage_summary,
)
from ._ntops.oot_support import linear
from ._ntops.torch.gelu import gelu as nt_gelu
from ._ntops.torch.rms_norm import rms_norm as nt_rms_norm

ensure_registered()
