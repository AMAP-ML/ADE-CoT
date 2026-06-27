import math

import torch
import torch.nn.functional as F
from xfuser.model_executor.layers.usp import USP

import sys 

try:
    import flash_attn
    from flash_attn.flash_attn_interface import (
        _flash_attn_forward,
        flash_attn_func,
        flash_attn_varlen_func,
    )
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None
    flash_attn_func = None

MEMORY_LAYOUT = {
    # flash:    expects [batch_size, seq_len, num_heads, head_dim]; no post-processing needed.
    # torch / vanilla / xdit: swap seq and head dims via [B,S,A,D] <-> [B,A,S,D].
    "flash": (
        lambda x: x,
        lambda x: x,
    ),
    "torch": (
        lambda x: x.transpose(1, 2),  # (B,S,A,D) -> (B,A,S,D)
        lambda x: x.transpose(1, 2),  # (B,A,S,D) -> (B,S,A,D)
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "xdit": (
        lambda x: x.transpose(1, 2),  # (B,S,A,D) -> (B,A,S,D)
        lambda x: x.transpose(1, 2),  # (B,A,S,D) -> (B,S,A,D)
    )
}


def attention(
    q,
    k,
    v,
    mode="flash",
    drop_rate=0,
    attn_mask=None,
    causal=False,
):
    """
    QKV self-attention.

    Args:
        q (torch.Tensor): query tensor of shape [batch_size, seq_len, num_heads, head_dim]
        k (torch.Tensor): key   tensor of shape [batch_size, seq_len_kv, num_heads, head_dim]
        v (torch.Tensor): value tensor of shape [batch_size, seq_len_kv, num_heads, head_dim]
        mode (str): attention backend, one of 'flash', 'torch', 'vanilla', 'xdit'
        drop_rate (float): dropout probability applied to the attention matrix
        attn_mask (torch.Tensor): optional attention mask (shape varies per backend)
        causal (bool): whether to use causal attention

    Returns:
        torch.Tensor: attention output of shape [batch_size, seq_len, num_heads * head_dim]
    """
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]

    q = pre_attn_layout(q)
    k = pre_attn_layout(k)
    v = pre_attn_layout(v)

    if mode == "torch":
        # PyTorch built-in scaled_dot_product_attention
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
        )
    elif mode == "flash":
        assert flash_attn_func is not None, "flash_attn_func is undefined"
        assert attn_mask is None, "attention mask is not supported in flash mode"
        x: torch.Tensor = flash_attn_func(
            q, k, v, dropout_p=drop_rate, causal=causal, softmax_scale=None
        )  # type: ignore
    elif mode == "vanilla":
        # Manual attention implementation
        scale_factor = 1 / math.sqrt(q.size(-1))  # 1 / sqrt(d_k)

        b, a, s, _ = q.shape
        s1 = k.size(2)  # kv sequence length

        # Attention bias buffer
        attn_bias = torch.zeros(b, a, s, s1, dtype=q.dtype, device=q.device)

        # Causal mask
        if causal:
            assert attn_mask is None, "causal mask and attn_mask cannot be combined"
            temp_mask = torch.ones(b, a, s, s, dtype=torch.bool, device=q.device).tril(
                diagonal=0
            )
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias = attn_bias.to(q.dtype)

        # Custom attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask  # allow ALiBi-style positional bias

        attn = (q @ k.transpose(-2, -1)) * scale_factor  # [B,A,S,S1]
        attn += attn_bias

        attn = attn.softmax(dim=-1)
        attn = torch.dropout(attn, p=drop_rate, train=True)

        x = attn @ v  # [B,A,S,D]
    elif mode == "xdit":
        x: torch.Tensor = USP(q, k, v, dropout_p=drop_rate, is_causal=causal)
    else:
        raise NotImplementedError(f"unsupported attention mode: {mode}")

    x = post_attn_layout(x)

    # Merge head dim
    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)  # [B,S,A*D]
    return out


def attention_with_cross_attn_score(
    q,
    k,
    v,
    mode="flash",
    drop_rate=0,
    attn_mask=None,
    causal=False,
):
    """
    QKV self-attention that also returns the per-head softmax LSE.
    """
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]

    q = pre_attn_layout(q)
    k = pre_attn_layout(k)
    v = pre_attn_layout(v)

    if mode == "flash":
        assert flash_attn_func is not None, "flash_attn_func is undefined"
        assert attn_mask is None, "attention mask is not supported in flash mode"

        x, softmax_lse, _ = flash_attn_func(
            q, k, v, dropout_p=drop_rate, causal=causal, softmax_scale=None, return_attn_probs=True, 
        )  # type: ignore

    else:
        raise NotImplementedError(f"unsupported attention mode: {mode}")

    x = post_attn_layout(x)

    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)  # [B,S,A*D]

    return out, softmax_lse
