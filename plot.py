# -*- coding : utf-8 -*-
"""Figure-3 style visualization for U-Net ResBlock features.

The script captures, for selected ResNet blocks in the U-Net up path:
  - h(x): the residual branch contribution reconstructed from the block output
  - f(x): the full ResBlock output feature

Unlike a point-cloud t-SNE scatter plot, this script renders feature maps as
spatial RGB images, matching Fig. 3 in the DiffArtist paper:
columns are diffusion noise levels and rows are noised image, h(x), and f(x).

Example:
    python plot.py \
        --config example_config.yaml \
        --image_dir data/example/2.png \
        --layers 0 \
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
    h_map: torch.Tensor
    f_map: torch.Tensor
    sample_labels: List[str]
    spatial_shape: Tuple[int, int]


@dataclass
class ResBlockFeatureCollector:
    """Collect h(x) and f(x) from ResnetBlock2D modules in U-Net up blocks."""

    pipe: object
    layers: Optional[Sequence[int]] = None
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

    @staticmethod
    def _flatten(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach().float().cpu()
        bsz, channels, height, width = tensor.shape
        return tensor.permute(0, 2, 3, 1).reshape(bsz, height * width, channels)

    @staticmethod
    def _to_map(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().float().cpu()

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
                h=self._flatten(h),
                f=self._flatten(output),
                h_map=self._to_map(h),
                f_map=self._to_map(output),
                sample_labels=self._sample_labels(output.shape[0]),
                spatial_shape=(output.shape[-2], output.shape[-1]),
            )
            print(f"Captured layer {layer_idx}: h/f shape {tuple(output.shape)}")

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
                            target.register_forward_hook(self._make_hook(name, layer_idx))
                        )
                    layer_idx += 1

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _load_pipe(cfg):
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if cfg.model == "sd":
        model_id = os.environ.get(
            "HF_MODEL_ID", cfg.get("model_id", "Manojb/stable-diffusion-2-1-base")
        )
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                token=hf_token,
            ).to(device)
        except OSError as exc:
            raise OSError(
                f"Could not load Stable Diffusion model {model_id!r}. "
                "If this is a gated Hugging Face repo, accept its license and set "
                "HF_TOKEN/HUGGINGFACE_TOKEN, or set model_id in the config to a "
                "local model directory or accessible repo."
            ) from exc
    elif cfg.model == "playground":
        model_id = os.environ.get(
            "HF_MODEL_ID", cfg.get("model_id", "playgroundai/playground-v2-1024px-aesthetic")
        )
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
            variant="fp16",
            token=hf_token,
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
def _decode_latent(pipe, latent: torch.Tensor) -> np.ndarray:
    original_dtype = next(iter(pipe.vae.post_quant_conv.parameters())).dtype
    pipe.vae.to(dtype=torch.float32)
    latent = latent.detach().to(device=device, dtype=torch.float32)
    decoded = pipe.vae.decode(latent / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image = (decoded / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
    pipe.vae.to(dtype=original_dtype)
    return image


def _step_from_noise_level(noise_level: float, num_steps: int, num_latents: int) -> int:
    noise_level = float(np.clip(noise_level, 0.0, 1.0))
    step = int(round((1.0 - noise_level) * (num_steps - 1)))
    return max(0, min(step, num_latents - 1))


@torch.no_grad()
def _capture_figure3_features(pipe, cfg, image, args):
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

    pipe.scheduler.set_timesteps(args.num_steps, device=device)
    added_cond_kwargs, text_embeddings = _empty_conditioning(pipe)

    captures = []
    for noise_level in args.noise_levels:
        capture_step = _step_from_noise_level(
            noise_level, len(pipe.scheduler.timesteps), len(inverted_latents)
        )
        timestep = pipe.scheduler.timesteps[capture_step]
        latent = inverted_latents[-(capture_step + 1)][None]
        noised_image = _decode_latent(pipe, latent)

        latent_model_input = torch.cat([latent] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

        collector = ResBlockFeatureCollector(pipe=pipe, layers=args.layers)
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

        captures.append(
            {
                "noise_level": noise_level,
                "capture_step": capture_step,
                "timestep": int(timestep.item()) if hasattr(timestep, "item") else int(timestep),
                "noised_image": noised_image,
                "records": collector.records,
            }
        )
    return captures


def _standardize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)


def _normalize_rgb(x: np.ndarray) -> np.ndarray:
    lo = np.percentile(x, 1, axis=(0, 1), keepdims=True)
    hi = np.percentile(x, 99, axis=(0, 1), keepdims=True)
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)


def _feature_map_to_tsne_rgb(
    feature_map: torch.Tensor,
    sample_index: int,
    perplexity: float,
    seed: int,
) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError(
            "plot.py needs scikit-learn for t-SNE. Install it with "
            "`pip install scikit-learn` in this environment."
        ) from exc

    sample_index = min(sample_index, feature_map.shape[0] - 1)
    feat = feature_map[sample_index]
    channels, height, width = feat.shape
    x = feat.permute(1, 2, 0).reshape(height * width, channels).numpy()
    perplexity = min(perplexity, max(2, x.shape[0] // 3))
    rgb = TSNE(
        n_components=3,
        perplexity=perplexity,
        init="pca",
        learning_rate=200.0,
        random_state=seed,
    ).fit_transform(_standardize(x))
    return _normalize_rgb(rgb.reshape(height, width, 3))


def _save_tensor_dump(captures, layer_name: str, out_path: str):
    torch.save(
        {
            "noise_levels": [item["noise_level"] for item in captures],
            "capture_steps": [item["capture_step"] for item in captures],
            "timesteps": [item["timestep"] for item in captures],
            "h": [item["records"][layer_name].h for item in captures],
            "f": [item["records"][layer_name].f for item in captures],
            "h_map": [item["records"][layer_name].h_map for item in captures],
            "f_map": [item["records"][layer_name].f_map for item in captures],
            "sample_labels": captures[0]["records"][layer_name].sample_labels,
            "spatial_shape": captures[0]["records"][layer_name].spatial_shape,
        },
        out_path,
    )


def save_figure3_plots(captures, out_dir: str, perplexity: float, seed: int, sample_index: int):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    if not captures or not captures[0]["records"]:
        raise RuntimeError("No ResBlock features were captured. Check --layers.")

    common_layers = set(captures[0]["records"].keys())
    for item in captures[1:]:
        common_layers &= set(item["records"].keys())
    if not common_layers:
        raise RuntimeError("No common ResBlock layer was captured across noise levels.")

    for layer_name in sorted(common_layers):
        fig, axes = plt.subplots(
            3,
            len(captures),
            figsize=(2.2 * len(captures), 5.2),
            dpi=180,
            squeeze=False,
        )
        for col, item in enumerate(captures):
            record = item["records"][layer_name]
            h_rgb = _feature_map_to_tsne_rgb(record.h_map, sample_index, perplexity, seed)
            f_rgb = _feature_map_to_tsne_rgb(record.f_map, sample_index, perplexity, seed)

            axes[0, col].imshow(item["noised_image"])
            axes[1, col].imshow(h_rgb)
            axes[2, col].imshow(f_rgb)
            axes[0, col].set_title(f"{item['noise_level']:.1f}T", fontsize=10)

            for row in range(3):
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                for spine in axes[row, col].spines.values():
                    spine.set_visible(False)

        axes[0, 0].set_ylabel("Noised\ncontent image", fontsize=10, rotation=90)
        axes[1, 0].set_ylabel("$h(x)$", fontsize=10, rotation=90)
        axes[2, 0].set_ylabel("$f(x)$", fontsize=10, rotation=90)
        fig.suptitle(f"{layer_name}: ResBlock feature map visualization", fontsize=11)
        fig.tight_layout(pad=0.25, rect=[0, 0, 1, 0.95])

        fig_path = os.path.join(out_dir, f"{layer_name}_figure3.png")
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        _save_tensor_dump(captures, layer_name, os.path.join(out_dir, f"{layer_name}_features.pt"))
        print(f"Saved {fig_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Figure-3 style t-SNE maps for U-Net ResBlock features")
    parser.add_argument("--config", type=str, default="example_config.yaml")
    parser.add_argument("--image_dir", type=str, default="data/example/1.png")
    parser.add_argument("--out_dir", type=str, default="out/tsne")
    parser.add_argument("--layers", type=int, nargs="*", default=[0])
    parser.add_argument("--noise_levels", type=float, nargs="*", default=[0.8, 0.6, 0.4, 0.2])
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
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

    captures = _capture_figure3_features(pipe, cfg, image, args)
    save_figure3_plots(
        captures,
        args.out_dir,
        args.perplexity,
        args.seed,
        args.sample_index,
    )


if __name__ == "__main__":
    main()
