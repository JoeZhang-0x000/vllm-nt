"""vllm-nt: NineToothed operator plugin for vLLM."""


def register() -> None:
    """Entry point called by vLLM's general_plugins system.

    Importing vllm_nt.oot triggers @register_oot decorators,
    which register our NineToothed operator implementations
    as OOT replacements for vLLM's built-in layers.
    """
    import vllm_nt.oot  # noqa: F401
