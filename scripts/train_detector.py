from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from stegastamp.data import Div2KDetectorDataset
from stegastamp.models import BiSeNetDetector
from stegastamp.utils import ensure_dir, get_git_hash, set_seed


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = ensure_dir(args.output_dir)

    ds = Div2KDetectorDataset(
        backgrounds_dir=args.backgrounds_dir,
        stamps_dir=args.stamps_dir,
        out_size=args.image_size,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    model = BiSeNetDetector().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logs: list[dict[str, float]] = []
    step = 0

    for epoch in range(args.epochs):
        model.train()
        for images, masks in dl:
            step += 1
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % args.log_every == 0:
                with torch.no_grad():
                    pred = (torch.sigmoid(logits) > 0.5).float()
                    iou = (pred * masks).sum() / ((pred + masks) > 0).float().sum().clamp_min(1.0)
                row = {"epoch": float(epoch), "step": float(step), "loss": float(loss.item()), "iou": float(iou.item())}
                logs.append(row)
                print(row)

        torch.save(
            {
                "detector": model.state_dict(),
                "git_hash": get_git_hash(),
                "cfg": vars(args),
                "epoch": epoch,
            },
            out_dir / f"detector_epoch_{epoch:03d}.pt",
        )

    with open(out_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train BiSeNet-style detector for StegaStamp regions.")
    p.add_argument("--backgrounds-dir", required=True)
    p.add_argument("--stamps-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
