from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from stegastamp.ecc import BCHCodec
from stegastamp.models import BiSeNetDetector, StegaStampDecoder
from stegastamp.pipeline import FullDecodePipeline
from stegastamp.utils import ensure_dir


def bitstring_to_array(s: str) -> np.ndarray:
    return np.asarray([1 if ch == "1" else 0 for ch in s.strip()], dtype=np.uint8)


def read_labels(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_capture_metadata(path: str | None) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    rows: dict[str, dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["image_name"]] = row
    return rows


def percentile_report(accs: list[float]) -> dict[str, float]:
    arr = np.asarray(accs, dtype=np.float32)
    if arr.size == 0:
        return {"p5": 0.0, "p25": 0.0, "p50": 0.0, "mean": 0.0}
    return {
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "mean": float(arr.mean()),
    }


def run(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(args.output_dir)
    device = torch.device(args.device)

    dec_ckpt = torch.load(args.decoder_checkpoint, map_location=device)
    detector_ckpt = torch.load(args.detector_checkpoint, map_location=device)
    message_bits = int(dec_ckpt["cfg"]["message_bits"])

    detector = BiSeNetDetector().to(device)
    detector.load_state_dict(detector_ckpt["detector"])
    decoder = StegaStampDecoder(message_bits=message_bits).to(device)
    decoder.load_state_dict(dec_ckpt["decoder"])
    bch = BCHCodec() if args.use_bch else None
    pipeline = FullDecodePipeline(detector=detector, decoder=decoder, message_bits=message_bits, bch=bch, device=args.device)

    labels = read_labels(args.labels_json)
    metadata = read_capture_metadata(args.capture_metadata_csv)
    image_paths = [p for p in Path(args.images_dir).glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    per_image = []
    grouped: dict[str, list[float]] = {}
    all_accs: list[float] = []

    for img_path in image_paths:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        decoded = pipeline.decode(rgb, detector_thresh=args.detector_thresh)
        if not decoded:
            per_image.append({"image_name": img_path.name, "detected": False})
            continue

        best = decoded[0]
        result = {
            "image_name": img_path.name,
            "detected": True,
            "confidence": float(best.confidence),
        }
        gt = labels.get(img_path.name)
        if gt is not None:
            gt_bits = bitstring_to_array(gt)[: message_bits]
            bit_acc = float((best.raw_bits[: gt_bits.size] == gt_bits).mean())
            result["raw_bit_accuracy"] = bit_acc
            all_accs.append(bit_acc)
            md = metadata.get(img_path.name, {})
            key = f"{md.get('camera', 'unknown')}|{md.get('media', 'unknown')}"
            grouped.setdefault(key, []).append(bit_acc)
        if best.corrected_bits is not None:
            result["bch_corrected_payload_bits"] = "".join(str(int(v)) for v in best.corrected_bits.tolist())
            result["bch_errors_corrected"] = int(best.n_corrected_errors or 0)
        per_image.append(result)

    with open(Path(out_dir) / "decode_results_per_image.json", "w", encoding="utf-8") as f:
        json.dump(per_image, f, indent=2)

    summary = {"overall": percentile_report(all_accs), "by_camera_media": {}}
    for k, vals in grouped.items():
        summary["by_camera_media"][k] = percentile_report(vals)
    with open(Path(out_dir) / "decode_summary_table.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate full detector+rectify+decode pipeline.")
    p.add_argument("--decoder-checkpoint", required=True)
    p.add_argument("--detector-checkpoint", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--labels-json", default=None, help="JSON mapping image_name -> bitstring ground truth.")
    p.add_argument("--capture-metadata-csv", default=None, help="CSV with columns: image_name,camera,media")
    p.add_argument("--use-bch", action="store_true")
    p.add_argument("--detector-thresh", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
