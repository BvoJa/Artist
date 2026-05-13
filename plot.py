# -*- coding : utf-8 -*-
"""Visualize U-Net ResBlock features with t-SNE.

The script captures, for selected ResNet blocks in the U-Net up path:
  - h(x): the residual branch contribution reconstructed from the block output
  - f(x): the full ResBlock output feature

Example:
    python plot.py \
        --config example_config.yaml \
        --image_dir data/example/1.png \
        --layers 0 1 2 3 \1
        --out_dir out/tsne
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.models import resnet
from omegaconf import OmegaConf

from injection_main import (
    _encode_text_sdxl_with_negative,
    device,
    invert,
)
import utils.exp_utils


@dataclass
class FeatureRecord:
    h: torch.Tensor
    f: torch.Tensor
    sample_labels: List[str]
    spatial_shape: Tuple[int, int]


@dataclass
class ResBlockFeatureCollector:
    """Collect h(x) and f(x) from ResnetBlock2D modules in U-Net up blocks."""

    pipe: object
    layers: Optional[Sequence[int]] = None
    max_tokens_per_sample: int = 512
    capture_once: bool = True
    records: Dict[str, FeatureRecord] = field(default_factory=dict)
    handles: List[torch.utils.hooks.RemovableHandle] = field(default_factory=list)

    def _selected(self, idx: int) -> bool:
        return self.layers is None or idx in self.layers

    def _sample_labels(self, batch_size: int) -> List[str]:
        if batch_size == 2:
            return ["uncond/image", "cond/empty"]
        return [f"sample/{i}" for i in range(batch_size)]

    @staticmethod
    def _shortcut(module: resnet.ResnetBlock2D, input_tensor: torch.Tensor) -> torch.Tensor:
        if getattr(module, "conv_shortcut", None) is not None:
            return module.conv_shortcut(input_tensor)
        return input_tensor

    def _flatten_and_sample(self, tensor: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, N, C], sampled equally per sample for readable t-SNE.
        tensor = tensor.detach().float().cpu()
        bsz, channels, height, width = tensor.shape
        tokens = tensor.permute(0, 2, 3, 1).reshape(bsz, height * width, channels)
        num_tokens = tokens.shape[1]
        if self.max_tokens_per_sample and num_tokens > self.max_tokens_per_sample:
            idx = torch.linspace(
                0, num_tokens - 1, steps=self.max_tokens_per_sample
            ).long()
            tokens = tokens[:, idx]
        return tokens

    def _make_hook(self, name: str, layer_idx: int):
        def hook(module, inputs, output):
            if self.capture_once and name in self.records:
                return
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                return

            input_tensor = inputs[0].detach()
            with torch.no_grad():
                shortcut = self._shortcut(module, input_tensor).detach()
                h = output.detach() * module.output_scale_factor - shortcut

            self.records[name] = FeatureRecord(
                h=self._flatten_and_sample(h),
                f=self._flatten_and_sample(output),
                sample_labels=self._sample_labels(output.shape[0]),
                spatial_shape=(output.shape[-2], output.shape[-1]),
            )
            print(
                f"Captured layer {layer_idx}: h/f shape "
                f"{tuple(output.shape)} -> {self.records[name].h.shape[1]} tokens/sample"
            )

        return hook

    def register(self):
        if isinstance(self.pipe, StableDiffusionPipeline):
            up_blocks = self.pipe.unet.up_blocks[1:]
        elif isinstance(self.pipe, StableDiffusionXLPipeline):
            up_blocks = self.pipe.unet.up_blocks[:-1]
        else:
            up_blocks = self.pipe.unet.up_blocks

        layer_idx = 0
        for block_idx, block in enumerate(up_blocks):
            for resnet_idx, module in enumerate(getattr(block, "resnets", [])):
                target = module.block if hasattr(module, "block") else module
                if isinstance(target, resnet.ResnetBlock2D):
                    if self._selected(layer_idx):
                        name = f"up{block_idx}_res{resnet_idx}_layer{layer_idx}"
                        self.handles.append(
                            target.register_forward_hook(
                                self._make_hook(name, layer_idx)
                            )
                        )
                    layer_idx += 1

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _load_pipe(cfg):
    if cfg.model == "sd":
        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base"
        ).to(device)
    elif cfg.model == "playground":
        pipe = DiffusionPipeline.from_pretrained(
            "playgroundai/playground-v2-1024px-aesthetic",
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
            variant="fp16",
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {cfg.model}")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe


def _empty_conditioning(pipe):
    if isinstance(pipe, StableDiffusionPipeline):
        return (
            None,
            pipe._encode_prompt(
                "",
                device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt="",
            ),
        )
    if isinstance(pipe, StableDiffusionXLPipeline):
        added_cond_kwargs, text_embeddings = _encode_text_sdxl_with_negative(pipe, [""])
        return added_cond_kwargs, text_embeddings
    raise TypeError(f"Unsupported pipeline type: {type(pipe)}")


@torch.no_grad()
def _capture_image_features(pipe, cfg, image, args, collector):
    with torch.no_grad():
        pipe.vae.to(dtype=torch.float32)
        latent = pipe.vae.encode(image.to(device) * 2 - 1)
        latents = pipe.vae.config.scaling_factor * latent.latent_dist.sample()
        if cfg.model == "playground":
            pipe.vae.to(dtype=torch.float16)

    inverted_latents = invert(
        pipe,
        latents,
        "",
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_steps,
    )

    capture_step = min(args.capture_step, len(inverted_latents) - 1)
    pipe.scheduler.set_timesteps(args.num_steps, device=device)
    timestep = pipe.scheduler.timesteps[capture_step]
    added_cond_kwargs, text_embeddings = _empty_conditioning(pipe)

    latent = inverted_latents[-(capture_step + 1)][None]
    latent_model_input = torch.cat([latent] * 2)
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

    collector.register()
    try:
        pipe.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=text_embeddings,
            added_cond_kwargs=added_cond_kwargs,
        )
    finally:
        collector.close()


def _standardize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)


def _plot_tsne(
    features: torch.Tensor,
    sample_labels: Sequence[str],
    title: str,
    out_path: str,
    perplexity: float,
    seed: int,
):
    import matplotlib.pyplot as plt

    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError(
            "plot.py needs scikit-learn for t-SNE. Install it with "
            "`pip install scikit-learn` in this environment."
        ) from exc

    bsz, tokens_per_sample, channels = features.shape
    x = features.reshape(bsz * tokens_per_sample, channels).numpy()
    labels = np.repeat(np.asarray(sample_labels), tokens_per_sample)

    # t-SNE requires perplexity < n_samples.
    perplexity = min(perplexity, max(2, x.shape[0] // 3))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate=200.0,
        random_state=seed,
    ).fit_transform(_standardize(x))

    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    for label in sample_labels:
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=7,
            alpha=0.6,
            linewidths=0,
            label=label,
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_plots(records: Dict[str, FeatureRecord], out_dir: str, perplexity: float, seed: int):
    os.makedirs(out_dir, exist_ok=True)
    if not records:
        raise RuntimeError("No ResBlock features were captured. Check --layers.")

    for name, record in records.items():
        for kind, tensor in [("h", record.h), ("f", record.f)]:
            out_path = os.path.join(out_dir, f"{name}_{kind}_tsne.png")
            title = f"{name} {kind}(x), spatial {record.spatial_shape[0]}x{record.spatial_shape[1]}"
            _plot_tsne(
                tensor,
                record.sample_labels,
                title,
                out_path,
                perplexity=perplexity,
                seed=seed,
            )
            print(f"Saved {out_path}")

        torch.save(
            {
                "h": record.h,
                "f": record.f,
                "sample_labels": record.sample_labels,
                "spatial_shape": record.spatial_shape,
            },
            os.path.join(out_dir, f"{name}_features.pt"),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="t-SNE plots for U-Net ResBlock features")
    parser.add_argument("--config", type=str, default="example_config.yaml")
    parser.add_argument("--image_dir", type=str, default="data/example/1.png")
    parser.add_argument("--out_dir", type=str, default="out/tsne")
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--max_tokens_per_sample", type=int, default=512)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument(
        "--capture_step",
        type=int,
        default=0,
        help="Denoising step to visualize after DDIM inversion. 0 uses the noisiest inverted latent.",
    )
    parser.add_argument("--guidance_scale", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    args.seed = cfg.seed if args.seed is None else args.seed
    args.num_steps = cfg.num_steps if args.num_steps is None else args.num_steps
    args.guidance_scale = (
        cfg.style_cfg_scale if args.guidance_scale is None else args.guidance_scale
    )

    utils.exp_utils.seed_all(args.seed)
    pipe = _load_pipe(cfg)
    image = utils.exp_utils.get_processed_image(args.image_dir, device, 512)

    collector = ResBlockFeatureCollector(
        pipe=pipe,
        layers=args.layers,
        max_tokens_per_sample=args.max_tokens_per_sample,
    )
    _capture_image_features(pipe, cfg, image, args, collector)
    save_plots(collector.records, args.out_dir, args.perplexity, args.seed)


if __name__ == "__main__":
    main()
