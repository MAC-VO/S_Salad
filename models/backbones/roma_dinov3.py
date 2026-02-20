"""
RoMaV2 DINOv3 backbone wrapper for SALAD training.

This module loads DINOv3 encoder weights from a local RoMaV2 checkpoint and
exposes SALAD-compatible outputs: feature map + CLS token.
"""

from pathlib import Path
import typing as T

import torch
import torch.nn as nn


ROMAV2_DINOV3_ARCHS = {
    "romav2_dinov3_vitl16": 1024,
}
_DINOV3_REPO = "facebookresearch/dinov3:adc254450203739c8149213a7a69d8d905b4fcfa"
_DINOV3_MODEL = "dinov3_vitl16"


def _extract_roma_descriptor_state_dict(checkpoint: dict[str, T.Any]) -> dict[str, torch.Tensor]:
    """
    Extract descriptor encoder weights from a RoMaV2 checkpoint.

    Args:
        checkpoint: Raw checkpoint dictionary containing RoMaV2 module weights.

    Returns:
        A state_dict matching DINOv3 encoder keys.
    """
    descriptor_state: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not key.startswith("f."):
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                "RoMaV2 descriptor state contains non-tensor weight: "
                f"{key} has type {type(value)}."
            )
        descriptor_state[key[2:]] = value

    if len(descriptor_state) == 0:
        raise ValueError(
            "RoMaV2 checkpoint does not contain descriptor keys with prefix 'f.'."
        )
    return descriptor_state


def _split_prefix_and_patch_tokens(
    tokens: torch.Tensor,
    *,
    num_patch_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split ViT output tokens into CLS token and patch tokens.

    DINOv3 can include extra prefix tokens (CLS + register tokens). We infer
    prefix length from the known patch-token count and keep the first prefix
    token as CLS.
    """
    if tokens.dim() != 3:
        raise ValueError(f"Expected tokens shape [B, N, C], got {tuple(tokens.shape)}.")
    if num_patch_tokens <= 0:
        raise ValueError(f"num_patch_tokens must be > 0, got {num_patch_tokens}.")

    _, token_count, _ = tokens.shape
    prefix_len = token_count - num_patch_tokens
    if prefix_len < 1:
        raise ValueError(
            "Expected at least one prefix token (CLS). "
            f"Got token_count={token_count}, num_patch_tokens={num_patch_tokens}."
        )

    cls_token = tokens[:, 0]
    patch_tokens = tokens[:, prefix_len:]
    return cls_token, patch_tokens


class RoMaDINOv3(nn.Module):
    """
    RoMaV2-initialized DINOv3 ViT-L/16 with configurable last-block finetuning.
    """

    def __init__(
        self,
        model_name: str = "romav2_dinov3_vitl16",
        romav2_ckpt_path: str = "",
        norm_layer: bool = True,
        return_token: bool = True,
        num_trainable_blocks: int = 0,
    ) -> None:
        super().__init__()

        if model_name not in ROMAV2_DINOV3_ARCHS:
            raise ValueError(
                f"Unknown model_name='{model_name}'. "
                f"Expected one of {list(ROMAV2_DINOV3_ARCHS.keys())}."
            )
        if len(romav2_ckpt_path) == 0:
            raise ValueError("romav2_ckpt_path is required.")
        if num_trainable_blocks < 0:
            raise ValueError(
                f"num_trainable_blocks must be >= 0, got {num_trainable_blocks}."
            )

        ckpt_path = Path(romav2_ckpt_path).expanduser()
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"RoMaV2 checkpoint not found at {ckpt_path}."
            )

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError(
                "RoMaV2 checkpoint payload must be a dict-like state_dict. "
                f"Got type={type(checkpoint)}."
            )

        descriptor_state = _extract_roma_descriptor_state_dict(checkpoint)

        self.model = torch.hub.load(
            repo_or_dir=_DINOV3_REPO,
            model=_DINOV3_MODEL,
            pretrained=False,
            skip_validation=True,
        )
        self.model.load_state_dict(descriptor_state, strict=True)

        self.num_trainable_blocks = int(num_trainable_blocks)
        num_blocks = len(self.model.blocks)
        if self.num_trainable_blocks > num_blocks:
            raise ValueError(
                "num_trainable_blocks exceeds the model block count: "
                f"got {self.num_trainable_blocks}, available {num_blocks}."
            )

        self.num_channels = ROMAV2_DINOV3_ARCHS[model_name]
        self.out_channels = self.num_channels
        self.patch_size = 16
        self.norm_layer = norm_layer
        self.return_token = return_token
        self._configure_trainable_params()

    def _configure_trainable_params(self) -> None:
        self.model.requires_grad_(False)
        if self.num_trainable_blocks > 0:
            for blk in self.model.blocks[-self.num_trainable_blocks:]:
                blk.requires_grad_(True)
            if self.norm_layer:
                self.model.norm.requires_grad_(True)

        trainable_count = sum(int(param.requires_grad) for param in self.model.parameters())
        if self.num_trainable_blocks == 0 and trainable_count != 0:
            raise RuntimeError("RoMaDINOv3 expected no trainable params when num_trainable_blocks=0.")
        if self.num_trainable_blocks > 0 and trainable_count == 0:
            raise RuntimeError(
                "RoMaDINOv3 expected trainable params when num_trainable_blocks>0."
            )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.num_trainable_blocks == 0:
            self.model.train(False)
        else:
            self.model.train(mode)
            if self.num_trainable_blocks < len(self.model.blocks):
                for blk in self.model.blocks[:-self.num_trainable_blocks]:
                    blk.train(False)
        return self

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor [B, 3, H, W], with H and W divisible by 16.

        Returns:
            If return_token=True:
                tuple(feature_map, cls_token)
                    feature_map: [B, C, H/16, W/16]
                    cls_token: [B, C]
            Else:
                feature_map: [B, C, H/16, W/16]
        """
        if x.dim() != 4:
            raise ValueError(f"Expected input shape [B, 3, H, W], got {tuple(x.shape)}.")
        batch_size, channels, height, width = x.shape
        if channels != 3:
            raise ValueError(f"Expected input channel size 3, got {channels}.")
        if (height % self.patch_size) != 0 or (width % self.patch_size) != 0:
            raise ValueError(
                f"Input spatial size must be divisible by {self.patch_size}, "
                f"got H={height}, W={width}."
            )

        tokens, (patch_h, patch_w) = self.model.prepare_tokens_with_masks(x)
        num_patch_tokens = patch_h * patch_w
        if self.num_trainable_blocks == 0:
            with torch.no_grad():
                for blk in self.model.blocks:
                    if self.model.rope_embed is not None:
                        rope_sincos = self.model.rope_embed(H=patch_h, W=patch_w)
                    else:
                        rope_sincos = None
                    tokens = blk(tokens, rope_sincos)
                if self.norm_layer:
                    tokens = self.model.norm(tokens)
        else:
            with torch.no_grad():
                for blk in self.model.blocks[:-self.num_trainable_blocks]:
                    if self.model.rope_embed is not None:
                        rope_sincos = self.model.rope_embed(H=patch_h, W=patch_w)
                    else:
                        rope_sincos = None
                    tokens = blk(tokens, rope_sincos)
            tokens = tokens.detach()
            for blk in self.model.blocks[-self.num_trainable_blocks:]:
                if self.model.rope_embed is not None:
                    rope_sincos = self.model.rope_embed(H=patch_h, W=patch_w)
                else:
                    rope_sincos = None
                tokens = blk(tokens, rope_sincos)
            if self.norm_layer:
                tokens = self.model.norm(tokens)

        cls_token, patch_tokens = _split_prefix_and_patch_tokens(
            tokens, num_patch_tokens=num_patch_tokens
        )
        features = patch_tokens.reshape(batch_size, patch_h, patch_w, self.num_channels)
        features = features.permute(0, 3, 1, 2).contiguous()

        if self.return_token:
            return features, cls_token
        return features
