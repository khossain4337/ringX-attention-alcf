# fused_attention is always available (no flash_attn dependency)
from .fused_attention import attention, HAS_FLASH

# These require flash_attn — skipped gracefully if not available
try:
    from .ringX1_attn import ringX1_attn_func
    from .ringX1o_attn import ringX1o_attn_func
    from .ringX2_attn import ringX2_attn_func
    from .ringX2o_attn import ringX2o_attn_func
    from .ringX3_attn import ringX3_attn_func
    from .ringX3b_attn import ringX3b_attn_func
    from .ringX4_attn import ringX4_attn_func
    from .ringX4o_attn import ringX4o_attn_func
except ImportError:
    pass

