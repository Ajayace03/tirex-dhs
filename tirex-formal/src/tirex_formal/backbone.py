"""TiRex-2 forward pass split into stages, with a per-series gate on variate attention.

TiRex-2 alternates a time mixer, which runs every series on its own, with a
variate mixer, which is attention across series. A covariate can only reach the
target through that attention, so the gate acts there: covariate i adds log g_i
to its attention logit as a key in every block. g_i = 1 is the unmodified
model, and g_i -> 0 is the same as deleting the series. Scaling the token
itself would not work, because the attention block RMS-normalises its input.

With no bias and every series kept, ``run_stack`` reproduces
``TiRex2._predict_once(..., tta_diff=False, pad_context=False)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Prepared:
    context: torch.Tensor
    target_mask: torch.Tensor
    known: torch.Tensor | None
    post_kwargs: dict


@dataclass
class Embedded:
    first_time: torch.Tensor
    scaler_state: tuple
    tokenizer_state: dict
    known: torch.Tensor | None
    context_patches: int


def prepare(backbone, target, covariates, prediction_length: int) -> Prepared:
    from tirex2 import TimeseriesType

    device = next(backbone.parameters()).device
    series = TimeseriesType(
        target=torch.as_tensor(target, dtype=torch.float32).view(1, -1),
        past_covariates=torch.as_tensor(covariates, dtype=torch.float32),
        future_covariates=None,
    )
    batch, _args, post_kwargs = backbone.postprocessor.transform_input(
        [series.target.to(device)],
        prediction_length,
        past_covariates=[series.past_covariates.to(device)],
        past_future_covariates=[None],
        tta_diff=False,
    )
    context = nn.functional.pad(
        batch["x"].to(device), (0, backbone.future_len - prediction_length), value=torch.nan
    )
    max_len = backbone.context_len + backbone.future_len
    if context.shape[-1] > max_len:
        context = context[..., -max_len:]
    target_mask = batch["target_mask"].to(device)
    known = None
    if context.shape[-1] >= backbone.future_len:
        has_future = ~torch.all(torch.isnan(context[:, -backbone.future_len:]), dim=-1)
        known = has_future & ~target_mask
    return Prepared(context, target_mask, known, post_kwargs)


def time_mix(block, x, target_mask, known):
    from tirex2.model.component.bi_xlstm import BiXLSTM
    from tirex2.model.component.variate_mixing_block import _unwrap_output

    if known is not None and isinstance(block.time_mixer, BiXLSTM):
        out = block.time_mixer(x, target_mask=target_mask, known_covariate_mask=known)
    else:
        out = block.time_mixer(x, target_mask=target_mask)
    return _unwrap_output(out)[0]


def embed(backbone, prepared: Prepared) -> Embedded:
    """Scale, patch, embed, and run the first time mixer on every series."""
    x, scaler_state = backbone.scaler.scale(prepared.context)
    x, tokenizer_state = backbone.tokenizer.input_transform(x)
    present = (~torch.isnan(x)).to(x.dtype)
    x = torch.nan_to_num(x, nan=backbone.nan_mask_value)
    x = backbone.input_patch_embedding(torch.cat((x, present), dim=-1))
    first = time_mix(backbone.stack[0], x, prepared.target_mask, prepared.known)
    future_patches = backbone.future_len // backbone.output_patch_size
    return Embedded(first, scaler_state, tokenizer_state, prepared.known, first.shape[1] - future_patches)


def variate_mix(mixer, x, group, target_mask, key_bias):
    """AttentionBlock.forward with an additive per-key bias.

    x is [rows, tokens, dim]. Rows in different groups never attend to each
    other, and covariates never attend to a target, as in TiRex-2.
    """
    xv = x.permute(1, 0, 2)
    _, inverse, counts = torch.unique(group, return_inverse=True, return_counts=True)
    keep = counts[inverse] > 1
    if not bool(keep.any()):
        return x
    sub = xv[:, keep]
    g = group[keep]
    tm = target_mask[keep]
    attn = mixer.attn
    tokens, rows, _ = sub.shape

    def heads(t):
        return t.reshape(tokens, rows, attn.n_heads, attn.kv_proj_dim).permute(0, 2, 1, 3)

    h = mixer.norm_attn(sub)
    q, k, v = heads(attn.WQ(h)), heads(attn.WK(h)), heads(attn.WV(h))
    if attn.use_qk_norm:
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
    allowed = (g.unsqueeze(0) == g.unsqueeze(1)) & ~(~tm.unsqueeze(1) & tm.unsqueeze(0))
    bias = torch.zeros(rows, rows, dtype=sub.dtype, device=sub.device).masked_fill(~allowed, float("-inf"))
    if key_bias is not None:
        bias = bias + key_bias[keep].to(sub.dtype).unsqueeze(0)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias[None, None], scale=attn.scale)
    out = attn.WO(out.permute(0, 2, 1, 3).reshape(tokens, rows, -1))
    sub = sub + out
    sub = sub + mixer.ffn(mixer.norm_ffn(sub))
    if bool(keep.all()):
        return sub.permute(1, 0, 2)
    full = xv.clone()
    full[:, keep] = sub
    return full.permute(1, 0, 2)


def run_stack(backbone, embedded: Embedded, rows, group, target_mask, key_bias, prediction_length: int):
    """Run the rows that are kept through the rest of TiRex-2.

    rows indexes the series of one prepared window and may repeat a series in
    several groups, one group per forecast horizon. Returns [len(rows), Q, T].
    """
    known = None if embedded.known is None else embedded.known.index_select(0, rows)
    x = embedded.first_time.index_select(0, rows)
    x = variate_mix(backbone.stack[0].variate_mixer, x, group, target_mask, key_bias)
    for block in list(backbone.stack)[1:]:
        x = time_mix(block, x, target_mask, known)
        x = variate_mix(block.variate_mixer, x, group, target_mask, key_bias)
    x = backbone.stack_out_norm(x)
    x = backbone.output_patch_embedding(x)
    x = torch.unflatten(x, -1, (backbone.num_quantiles, backbone.output_patch_size))
    x = torch.transpose(x, 1, 2)
    x = backbone.tokenizer.output_transform(x, embedded.tokenizer_state)
    scaler_state = tuple(None if part is None else part.index_select(0, rows) for part in embedded.scaler_state)
    x = backbone.scaler.re_scale(x, scaler_state)
    x = x[..., :-backbone.output_patch_size][:, :, -backbone.future_len:]
    return x[:, :, :prediction_length]


def decode_target(backbone, prepared: Prepared, target_row: torch.Tensor, n_series: int) -> torch.Tensor:
    """Apply TiRex-2's output calibration to one target row. Returns [Q, T]."""
    full = target_row.new_zeros(n_series, *target_row.shape)
    full = full.index_copy(0, torch.zeros(1, dtype=torch.long, device=full.device), target_row.unsqueeze(0))
    out = backbone.postprocessor.transform_output(
        full, preserve_grad=torch.is_grad_enabled(), **prepared.post_kwargs
    )
    return out[0][0]
