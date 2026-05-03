from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PerturbationConfig:
    perspective: bool = True
    blur: bool = True
    color: bool = True
    noise: bool = True
    jpeg: bool = True


@dataclass
class TrainConfig:
    image_size: int = 400
    message_bits: int = 100
    batch_size: int = 4
    epochs: int = 30
    lr_encoder_decoder: float = 1e-4
    lr_critic: float = 1e-4
    lambda_message: float = 1.0
    lambda_residual_max: float = 2.0
    lambda_lpips_max: float = 1.0
    lambda_critic_max: float = 0.1
    warmup_epochs_decode_only: int = 3
    perspective_ramp_factor: float = 0.6
    num_workers: int = 2
    seed: int = 42


def perturbation_profile(name: str) -> PerturbationConfig:
    """
    Paper ablation variants:
    - none: no perturbations
    - pixelwise: color + noise + jpeg
    - spatial: perspective + blur
    - all: all perturbations
    """
    key = name.strip().lower()
    if key == "none":
        return PerturbationConfig(False, False, False, False, False)
    if key == "pixelwise":
        return PerturbationConfig(False, False, True, True, True)
    if key == "spatial":
        return PerturbationConfig(True, True, False, False, False)
    if key == "all":
        return PerturbationConfig(True, True, True, True, True)
    raise ValueError(f"Unknown perturbation profile: {name}")
