from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torchvision import transforms

from .ecc import BCHCodec
from .geometry import mask_to_quads, warp_quad_to_square
from .models import BiSeNetDetector, StegaStampDecoder


@dataclass
class DecodeResult:
    raw_bits: np.ndarray
    corrected_bits: np.ndarray | None
    n_corrected_errors: int | None
    quad_xy: np.ndarray
    confidence: float


class FullDecodePipeline:
    """
    full_image -> detector mask -> quad proposals -> rectified crop -> decoder bits -> optional BCH correction
    """

    def __init__(
        self,
        detector: BiSeNetDetector,
        decoder: StegaStampDecoder,
        message_bits: int = 100,
        bch: BCHCodec | None = None,
        device: str = "cpu",
    ) -> None:
        self.detector = detector.to(device).eval()
        self.decoder = decoder.to(device).eval()
        self.message_bits = message_bits
        self.bch = bch
        self.device = torch.device(device)
        self.to_tensor = transforms.ToTensor()

    @torch.no_grad()
    def decode(self, image_rgb: np.ndarray, detector_thresh: float = 0.5) -> list[DecodeResult]:
        h, w = image_rgb.shape[:2]
        x = self.to_tensor(image_rgb).unsqueeze(0).to(self.device)
        mask_logits = self.detector(x)
        mask_prob = torch.sigmoid(mask_logits).squeeze().cpu().numpy()

        quads = mask_to_quads(mask_prob, thresh=detector_thresh)
        results: list[DecodeResult] = []
        for q in quads:
            crop = warp_quad_to_square(image_rgb, q.quad_xy, out_size=400)
            crop_t = self.to_tensor(crop).unsqueeze(0).to(self.device)
            logits = self.decoder(crop_t)
            raw_bits = (logits.sigmoid().squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)

            corrected_bits = None
            n_corr = None
            if self.bch is not None:
                codeword = self.bch.extract_from_message_bits(raw_bits)
                corrected_bits, n_corr = self.bch.decode_codeword_bits(codeword)

            results.append(
                DecodeResult(
                    raw_bits=raw_bits,
                    corrected_bits=corrected_bits,
                    n_corrected_errors=n_corr,
                    quad_xy=q.quad_xy,
                    confidence=q.score,
                )
            )
        return results
