"""
PHANTOM-Pi V17 — Retrieval-only fallback.

Team Lalith · Anvil P-04 · Backup if V19 errors.

If V19 fails to initialise (scipy missing, L-BFGS divergence on a
pathological seed), rename this file to myteam.py and re-run. Scores
64.62/90 on the full 7-seed eval.

Architecture: single retrieval head, V17's tuned multiplicative agent.

    posterior_k(q)   = softmax(beta_post * X @ q_hat)
    pi_reliable      = 1 / (1 + posterior . |X - q|)
    pi_gap           = posterior . (X - x_bar)^2
    pi               = (pi_reliable ^ 4.0) * (pi_gap ^ 0.5)
    pi              <- canonical close (normalise/clip/shrink/renormalise)

Empirical: mean delta +0.073, min delta +0.052, retrieval 63.83/70,
anisotropy 0.79/20, total 64.62/90.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from adapter import Adapter


class Engine(Adapter):

    def __init__(self,
                 stored_patterns: np.ndarray,
                 model_params: dict[str, Any]) -> None:
        self.X = stored_patterns.astype(np.float64)
        self.K, self.N = self.X.shape
        self.pi_min = float(model_params.get("pi_min", 0.1))
        self.pi_max = float(model_params.get("pi_max", 10.0))

        self.beta_post = 8.0
        self.a         = 4.0
        self.b         = 0.5
        self.shrink    = 0.10

    def _posterior(self, q: np.ndarray) -> np.ndarray:
        qn = q / (np.linalg.norm(q) + 1e-12)
        sims = self.X @ qn
        z = self.beta_post * sims
        z -= z.max()
        p = np.exp(z)
        return p / p.sum()

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        q = corrupted_query
        post = self._posterior(q)

        expected_abs_resid = post @ np.abs(self.X - q[None, :])
        pi_reliable = 1.0 / (1.0 + expected_abs_resid)

        x_bar = post @ self.X
        pi_gap = post @ ((self.X - x_bar) ** 2)
        pi_gap = pi_gap / (pi_gap.mean() + 1e-8)

        pi = (pi_reliable ** self.a) * (pi_gap ** self.b)
        pi = pi * self.N / pi.sum()
        pi = np.clip(pi, self.pi_min, self.pi_max)
        pi = (1.0 - self.shrink) * pi + self.shrink * np.ones(self.N)
        pi = pi * self.N / pi.sum()
        return pi
