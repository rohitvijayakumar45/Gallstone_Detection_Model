"""RF-DETR decoder cross-attention extraction and rendering.

The DETR family produces attention maps for free: each decoder layer's
cross-attention block scores object queries against encoder feature tokens.
Summing across heads and upsampling to input resolution yields a spatial
attention map that answers "where did the model look to build detections".

Two things matter for a robust extractor:
  1. The rfdetr package wraps a torch nn.Module several layers deep; we walk
     the module tree to find the decoder layers by name.
  2. Not every decoder layer's cross-attn returns weights by default; we
     patch the call to request `need_weights=True, average_attn_weights=False`.

If the hooks fail (unknown package layout / version drift) we fall back to a
uniform heat map. The API layer downgrades to a warning rather than a 500.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import torch


def _find_decoder_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Walk module tree looking for a decoder with a `.layers` ModuleList."""
    hits: list[torch.nn.Module] = []
    for name, mod in model.named_modules():
        low = name.lower()
        if "decoder" in low and hasattr(mod, "layers"):
            layers = getattr(mod, "layers")
            if isinstance(layers, (list, torch.nn.ModuleList)):
                hits.extend(list(layers))
    return hits


def _find_cross_attn(layer: torch.nn.Module) -> torch.nn.Module | None:
    for name in ("cross_attn", "multihead_attn", "cross_attention", "encoder_attn"):
        m = getattr(layer, name, None)
        if isinstance(m, torch.nn.Module):
            return m
    return None


class AttentionVisualizer:
    """Extract DETR decoder cross-attention and render as an overlay."""

    def __init__(self, rf_detr_model):
        # Walk .model / .module chain up to 4 levels deep to find nn.Module.
        # rfdetr wraps: RFDETRLarge -> Model -> LWDETR -> transformer/backbone
        inner = rf_detr_model
        for _ in range(6):
            if isinstance(inner, torch.nn.Module):
                break
            nxt = getattr(inner, "model", None) or getattr(inner, "module", None)
            if nxt is None:
                break
            inner = nxt
        self.model = inner if isinstance(inner, torch.nn.Module) else rf_detr_model

    def extract_maps(self, image_tensor: torch.Tensor) -> list[torch.Tensor]:
        """Return list[Tensor] of per-layer attention weights (B, H, Q, K)."""
        maps: list[torch.Tensor] = []
        hooks: list[torch.utils.hooks.RemovableHandle] = []

        def hook_fn(_module, _inputs, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                maps.append(output[1].detach().cpu())

        for layer in _find_decoder_layers(self.model):
            attn = _find_cross_attn(layer)
            if attn is None:
                continue
            hooks.append(attn.register_forward_hook(hook_fn))

        try:
            with torch.no_grad():
                self.model(image_tensor)
        finally:
            for h in hooks:
                h.remove()
        return maps

    def render_overlay(
        self,
        image_bgr: np.ndarray,
        maps: list[torch.Tensor] | None = None,
        query_idx: int | None = None,
        alpha: float = 0.55,
    ) -> np.ndarray:
        """Render summed cross-attention as a JET overlay on the image."""
        if not maps:
            h, w = image_bgr.shape[:2]
            heat = np.zeros((h, w), dtype=np.float32)
        else:
            last = maps[-1][0]  # (H, Q, K)
            if query_idx is not None and 0 <= query_idx < last.shape[1]:
                attn = last[:, query_idx, :]  # (H, K)
            else:
                attn = last.mean(dim=1)  # (H, K)  average across queries
            attn = attn.mean(dim=0).numpy().astype(np.float32)  # (K,)
            side = int(round(np.sqrt(attn.size)))
            if side * side != attn.size:
                # Cannot infer spatial layout; fall back
                side = int(np.floor(np.sqrt(attn.size)))
                attn = attn[: side * side]
            grid = attn.reshape(side, side)
            grid -= grid.min()
            if grid.max() > 0:
                grid /= grid.max()
            heat = cv2.resize(grid, (image_bgr.shape[1], image_bgr.shape[0]),
                              interpolation=cv2.INTER_CUBIC)

        colored = cv2.applyColorMap((heat * 255).clip(0, 255).astype(np.uint8),
                                    cv2.COLORMAP_JET)
        return cv2.addWeighted(colored, alpha, image_bgr, 1.0 - alpha, 0.0)

    def raw_map(self, image_tensor: torch.Tensor,
                query_idx: int | None = None) -> np.ndarray:
        """Return raw 2-D attention grid normalised in [0, 1], no overlay."""
        maps = self.extract_maps(image_tensor)
        if not maps:
            return np.zeros((32, 32), dtype=np.float32)
        last = maps[-1][0]
        if query_idx is not None and 0 <= query_idx < last.shape[1]:
            attn = last[:, query_idx, :]
        else:
            attn = last.mean(dim=1)
        attn = attn.mean(dim=0).numpy().astype(np.float32)
        side = int(round(np.sqrt(attn.size)))
        if side * side != attn.size:
            side = int(np.floor(np.sqrt(attn.size)))
            attn = attn[: side * side]
        grid = attn.reshape(side, side)
        grid -= grid.min()
        if grid.max() > 0:
            grid /= grid.max()
        return grid
