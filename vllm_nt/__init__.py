import os

if os.environ.get("VLLM_NT_ENABLE_CUSTOM_OP_REGISTER_INTERCEPT") == "1":
    __import__("vllm_nt.oot")

register = lambda: __import__("vllm_nt.oot")
