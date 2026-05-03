from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from stegastamp.data import MirflickrDataset
from stegastamp.models import StegaStampDecoder, StegaStampEncoder
from stegastamp.perturbations import PhysicalPerturbationPipeline
from stegastamp.utils import ensure_dir, set_seed


def sample_messages(batch_size: int, bits: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, 2, (batch_size, bits), device=device, dtype=torch.float32)


def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = ensure_dir(args.output_dir)

    ckpt = torch.load(args.checkpoint, map_location=device)
    msg_bits = int(ckpt["cfg"]["message_bits"])
    image_size = int(ckpt["cfg"]["image_size"])
    encoder = StegaStampEncoder(message_bits=msg_bits).to(device)
    decoder = StegaStampDecoder(message_bits=msg_bits).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()
    perturb = PhysicalPerturbationPipeline(enable=True).to(device)

    ds = MirflickrDataset(root=args.data_dir, image_size=image_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    strengths = [0.0, 0.5, 1.0, 1.5, 2.0]
    perturb_types = ["warp", "blur", "noise", "color", "jpeg"]
    metrics: dict[str, dict[str, float]] = {}

    with torch.no_grad():
        for p_name in perturb_types:
            for s in strengths:
                all_acc = []
                seen = 0
                for images in dl:
                    images = images.to(device)
                    messages = sample_messages(images.size(0), msg_bits, device)
                    encoded = encoder(images, messages).stegastamp
                    kwargs = {
                        "perspective_strength": s if p_name == "warp" else 0.0,
                        "blur_strength": s if p_name == "blur" else 0.0,
                        "noise_strength": s if p_name == "noise" else 0.0,
                        "color_strength": s if p_name == "color" else 0.0,
                        "jpeg_strength": s if p_name == "jpeg" else 0.0,
                    }
                    corrupted = perturb(encoded, **kwargs)
                    logits = decoder(corrupted)
                    batch_acc = ((logits.sigmoid() > 0.5) == messages.bool()).float().mean(dim=1).cpu().numpy()
                    all_acc.extend(batch_acc.tolist())
                    seen += images.size(0)
                    if seen >= args.max_samples:
                        break
                arr = np.asarray(all_acc, dtype=np.float32)
                key = f"{p_name}@{s:.1f}"
                metrics[key] = {
                    "mean": float(arr.mean()),
                    "p25": float(np.percentile(arr, 25)),
                    "p50": float(np.percentile(arr, 50)),
                    "p75": float(np.percentile(arr, 75)),
                }
                print(key, metrics[key])

    with open(Path(out_dir) / "synthetic_ablation.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Synthetic ablation evaluator for perturbation robustness.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
