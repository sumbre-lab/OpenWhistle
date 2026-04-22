from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2FeedForward
from typing import Optional ,Tuple
import torch.nn as nn
import torch
import xformers.ops as xops
from xformers.components.attention.core import scaled_dot_product_attention


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->Wav2Vec2
class Wav2Vec_FastAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper
    Efficient implementation using xformers
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        self.attn_drop = nn.Dropout(dropout, inplace=False)

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, tgt_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, tgt_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, tgt_len, self.num_heads, self.head_dim)

        # inputs must be [B,M,H,K]
        attn_output = xops.memory_efficient_attention(
            q, k, v,
            # attn_bias=attention_mask[:,:,:q.shape[1],:q.shape[1]],#attention_mask
            # p = self.dropout,
            scale = self.scaling,
            ) #op = xops.MemoryEfficientAttentionTritonFwdFlashBwOp
        ## Note that this is slower than using a non-materialized mask of xformers,
        # but allows us to use our custom bias and apply backpropagate the loss.
        # Key: the attention_mask stride need to be multiple of 8
        # in order to do that we cut the mask at the last minute, making
        # sure no strange stuff are allocated in between !
        # this did not resolve some memory leakage :(

        attn_output = attn_output.view(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output



class Wav2Vec_FastEncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Wav2Vec_FastAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):

        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.attention(
            hidden_states, attention_mask=attention_mask
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        outputs = (hidden_states,)

        return outputs




class Wav2Vec2_FastEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Wav2Vec_FastAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states  = self.attention(
            hidden_states, attention_mask=attention_mask
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        # if output_attentions:
        #     outputs += (attn_weights,)

        return outputs