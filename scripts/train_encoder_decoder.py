from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from stegastamp.config import TrainConfig, perturbation_profile
from stegastamp.data import MirflickrDataset
from stegastamp.ecc import BCHCodec
from stegastamp.losses import LPIPSLoss, ResidualLoss, WassersteinCriticLoss
from stegastamp.models import StegaStampCritic, StegaStampDecoder, StegaStampEncoder
from stegastamp.perturbations import PhysicalPerturbationPipeline
from stegastamp.utils import ensure_dir, get_git_hash, set_seed


def sample_random_messages(batch_size: int, bits: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, 2, (batch_size, bits), device=device, dtype=torch.float32)


def sample_bch_messages(batch_size: int, bits: int, bch: BCHCodec, device: torch.device) -> torch.Tensor:
    messages = []
    payload_bits = bch.cfg.data_bytes * 8
    for _ in range(batch_size):
        payload = np.random.randint(0, 2, payload_bits, dtype=np.uint8)
        codeword = bch.encode_payload_bits(payload)
        msg = bch.fit_to_message_bits(codeword, message_bits=bits)
        messages.append(msg)
    return torch.from_numpy(np.stack(messages).astype(np.float32)).to(device)


def train(args: argparse.Namespace) -> None:
    cfg = TrainConfig(
        image_size=args.image_size,
        message_bits=args.message_bits,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr_encoder_decoder=args.lr_encoder_decoder,
        lr_critic=args.lr_critic,
        lambda_message=args.lambda_message,
        lambda_residual_max=args.lambda_residual_max,
        lambda_lpips_max=args.lambda_lpips_max,
        lambda_critic_max=args.lambda_critic_max,
        warmup_epochs_decode_only=args.warmup_epochs_decode_only,
        perspective_ramp_factor=args.perspective_ramp_factor,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    set_seed(cfg.seed)
    p_cfg = perturbation_profile(args.perturbation_profile)
    out_dir = ensure_dir(args.output_dir)
    device = torch.device(args.device)

    ds = MirflickrDataset(args.data_dir, image_size=cfg.image_size)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )

    encoder = StegaStampEncoder(message_bits=cfg.message_bits).to(device)
    decoder = StegaStampDecoder(message_bits=cfg.message_bits).to(device)
    critic = StegaStampCritic().to(device)
    perturb = PhysicalPerturbationPipeline(enable=True).to(device)
    bce = nn.BCEWithLogitsLoss()
    residual_loss = ResidualLoss()
    lpips_loss = LPIPSLoss().to(device)
    critic_loss = WassersteinCriticLoss()
    opt_ed = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=cfg.lr_encoder_decoder)
    opt_c = optim.Adam(critic.parameters(), lr=cfg.lr_critic)
    bch = BCHCodec() if args.use_bch else None

    train_log: list[dict[str, float]] = []
    step = 0
    for epoch in range(cfg.epochs):
        encoder.train()
        decoder.train()
        critic.train()

        if epoch < cfg.warmup_epochs_decode_only:
            ramp = 0.0
        else:
            denom = max(1, cfg.epochs - cfg.warmup_epochs_decode_only - 1)
            ramp = (epoch - cfg.warmup_epochs_decode_only) / denom

        lambda_residual = cfg.lambda_residual_max * ramp
        lambda_lpips = cfg.lambda_lpips_max * ramp
        lambda_critic = cfg.lambda_critic_max * ramp
        persp_s = ramp * cfg.perspective_ramp_factor if p_cfg.perspective else 0.0
        other_s = ramp
        for images in dl:
            step += 1
            images = images.to(device)
            if bch is None:
                messages = sample_random_messages(images.size(0), cfg.message_bits, device)
            else:
                messages = sample_bch_messages(images.size(0), cfg.message_bits, bch, device)

            enc_out = encoder(images, messages)
            distorted = perturb(
                enc_out.stegastamp,
                perspective_strength=persp_s,
                blur_strength=other_s if p_cfg.blur else 0.0,
                color_strength=other_s if p_cfg.color else 0.0,
                noise_strength=other_s if p_cfg.noise else 0.0,
                jpeg_strength=other_s if p_cfg.jpeg else 0.0,
            )
            message_logits = decoder(distorted)

            # Critic step (interleaved)
            with torch.no_grad():
                encoded_detached = enc_out.stegastamp.detach()
            c_real = critic(images)
            c_fake = critic(encoded_detached)
            loss_c = critic_loss(c_real, c_fake)
            opt_c.zero_grad(set_to_none=True)
            loss_c.backward()
            opt_c.step()

            # Encoder/decoder step
            c_fake_for_gen = critic(enc_out.stegastamp)
            loss_msg = bce(message_logits, messages)
            loss_res = residual_loss(enc_out.residual)
            loss_lp = lpips_loss(enc_out.stegastamp, images)
            loss_ed = cfg.lambda_message * loss_msg + lambda_residual * loss_res + lambda_lpips * loss_lp + lambda_critic * (
                -c_fake_for_gen.mean()
            )
            opt_ed.zero_grad(set_to_none=True)
            loss_ed.backward()
            opt_ed.step()

            if step % args.log_every == 0:
                with torch.no_grad():
                    bit_acc = ((message_logits.sigmoid() > 0.5) == messages.bool()).float().mean().item()
                row = {
                    "epoch": float(epoch),
                    "step": float(step),
                    "loss_ed": float(loss_ed.item()),
                    "loss_msg": float(loss_msg.item()),
                    "loss_res": float(loss_res.item()),
                    "loss_lpips": float(loss_lp.item()),
                    "loss_critic": float(loss_c.item()),
                    "bit_acc": float(bit_acc),
                    "ramp": float(ramp),
                }
                train_log.append(row)
                print(row)

        ckpt = {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "critic": critic.state_dict(),
            "cfg": cfg.__dict__,
            "perturbation_profile": args.perturbation_profile,
            "ramp_epoch": epoch,
            "git_hash": get_git_hash(),
            "use_bch": args.use_bch,
        }
        torch.save(ckpt, out_dir / f"checkpoint_epoch_{epoch:03d}.pt")

    with open(out_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train StegaStamp encoder/decoder/critic.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--perturbation-profile", default="all", choices=["none", "pixelwise", "spatial", "all"])
    parser.add_argument("--use-bch", action="store_true")
    parser.add_argument("--image-size", type=int, default=400)
    parser.add_argument("--message-bits", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr-encoder-decoder", type=float, default=1e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-4)
    parser.add_argument("--lambda-message", type=float, default=1.0)
    parser.add_argument("--lambda-residual-max", type=float, default=2.0)
    parser.add_argument("--lambda-lpips-max", type=float, default=1.0)
    parser.add_argument("--lambda-critic-max", type=float, default=0.1)
    parser.add_argument("--warmup-epochs-decode-only", type=int, default=3)
    parser.add_argument("--perspective-ramp-factor", type=float, default=0.6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
