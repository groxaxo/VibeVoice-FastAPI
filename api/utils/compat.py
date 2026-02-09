"""Compatibility layer for different transformers versions."""

from typing import Optional

# Try to import FlashAttentionKwargs from transformers 4.51.3+
try:
    from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
except ImportError:
    # Fallback for older transformers versions
    from dataclasses import dataclass

    @dataclass
    class FlashAttentionKwargs:
        """Compatibility stub for FlashAttentionKwargs in older transformers versions."""

        # Add common flash attention kwargs as needed
        attention_mask: Optional = None
        causal: Optional = None
        alibi_bias: Optional = None
        alibi_bias_max: Optional = None
        use_flash_attention: Optional = None
        use_flash_attention_2: Optional = None


__all__ = ["FlashAttentionKwargs"]
