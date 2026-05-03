from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass
class DetectedQuad:
    quad_xy: np.ndarray  # shape [4,2] in image coordinates
    score: float


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Return points in TL, TR, BR, BL order.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def mask_to_quads(mask_prob: np.ndarray, thresh: float = 0.5, min_area: int = 600) -> list[DetectedQuad]:
    """
    Convert detector mask probabilities into quadrilateral proposals:
    connected components -> convex hull -> polygon approx -> fallback min area rect.
    """
    mask = (mask_prob >= thresh).astype(np.uint8) * 255
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    quads: list[DetectedQuad] = []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(hull)
            quad = cv2.boxPoints(rect).astype(np.float32)
        quad = order_quad_points(quad)
        score = float(mask_prob[comp > 0].mean()) if np.any(comp > 0) else 0.0
        quads.append(DetectedQuad(quad_xy=quad, score=score))
    quads.sort(key=lambda q: q.score, reverse=True)
    return quads


def warp_quad_to_square(image_bgr_or_rgb: np.ndarray, quad_xy: Sequence[Sequence[float]], out_size: int = 400) -> np.ndarray:
    quad = order_quad_points(np.asarray(quad_xy, dtype=np.float32))
    dst = np.array([[0, 0], [out_size - 1, 0], [out_size - 1, out_size - 1], [0, out_size - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(image_bgr_or_rgb, H, (out_size, out_size), flags=cv2.INTER_LINEAR)
