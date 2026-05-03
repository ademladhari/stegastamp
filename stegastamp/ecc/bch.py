from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import bchlib  # type: ignore
except Exception:  # pragma: no cover
    bchlib = None


@dataclass
class BCHConfig:
    poly: int = 137
    bits: int = 5
    data_bytes: int = 7  # 56 bits practical payload in paper


class BCHCodec:
    """
    BCH wrapper for paper-style '56 corrected bits from 100 raw bits' workflow.
    """

    def __init__(self, cfg: BCHConfig | None = None) -> None:
        if bchlib is None:
            raise RuntimeError("bchlib is required. Install with `pip install bchlib`.")
        self.cfg = cfg or BCHConfig()
        self.bch = bchlib.BCH(self.cfg.poly, self.cfg.bits)
        self.n_parity = self.bch.ecc_bytes
        self.total_bytes = self.cfg.data_bytes + self.n_parity
        self.total_bits = self.total_bytes * 8

    def encode_payload_bits(self, payload_bits: np.ndarray) -> np.ndarray:
        payload_bits = np.asarray(payload_bits, dtype=np.uint8).reshape(-1)
        if payload_bits.size != self.cfg.data_bytes * 8:
            raise ValueError(f"Expected {self.cfg.data_bytes * 8} payload bits, got {payload_bits.size}")
        payload = np.packbits(payload_bits).tobytes()
        ecc = self.bch.encode(payload)
        codeword = np.frombuffer(payload + ecc, dtype=np.uint8)
        return np.unpackbits(codeword)

    def decode_codeword_bits(self, codeword_bits: np.ndarray) -> tuple[np.ndarray, int]:
        codeword_bits = np.asarray(codeword_bits, dtype=np.uint8).reshape(-1)
        if codeword_bits.size != self.total_bits:
            raise ValueError(f"Expected {self.total_bits} bits, got {codeword_bits.size}")
        codeword = np.packbits(codeword_bits).tobytes()
        data = bytearray(codeword[: self.cfg.data_bytes])
        ecc = bytearray(codeword[self.cfg.data_bytes :])
        n_corr = self.bch.decode(data=data, recv_ecc=ecc)
        payload_bits = np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8))
        return payload_bits, int(n_corr)

    def fit_to_message_bits(self, codeword_bits: np.ndarray, message_bits: int = 100) -> np.ndarray:
        """
        Pack BCH codeword into fixed model message bit budget.
        """
        codeword_bits = np.asarray(codeword_bits, dtype=np.uint8).reshape(-1)
        if codeword_bits.size > message_bits:
            raise ValueError(f"Codeword ({codeword_bits.size}) longer than message_bits ({message_bits})")
        out = np.zeros(message_bits, dtype=np.uint8)
        out[: codeword_bits.size] = codeword_bits
        return out

    def extract_from_message_bits(self, model_bits: np.ndarray) -> np.ndarray:
        model_bits = np.asarray(model_bits, dtype=np.uint8).reshape(-1)
        return model_bits[: self.total_bits]
