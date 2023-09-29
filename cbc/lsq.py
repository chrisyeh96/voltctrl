"""Convex body chasing via projection."""
from __future__ import annotations

from typing import Any

import numpy as np

from cbc.base import CBCBase


class CBCLsq(CBCBase):
    """Returns the unconstrained least-squares estimate of X.

    Does NOT actually perform any convex body chasing.
    """
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.prev_t = -100

    def select(self, t: int) -> np.ndarray:
        """Selects a model.

        When select() is called, we have seen t observations. That is, we have values for:
          v(0), ...,   v(t)    # recall:   v(t) = v[t]
        q^c(0), ..., q^c(t)    # recall: q^c(t) = q[t]
          u(0), ...,   u(t-1)  # recall:   u(t) = u[t]
         Δv(0), ...,  Δv(t-1)  # recall:  Δv(t) = Δv[t]

        Args
        - t: int, current time step
        """
        if t == 0:
            return self.X_init

        if t != self.prev_t + 1:
            print(f'resetting A,B: prev_t={self.prev_t}, t={t}')
            self.A = np.array([np.outer(self.Δv[i], self.u[i]) for i in range(t)]).sum(axis=0)
            self.B = np.array([np.outer(self.u[i], self.u[i]) for i in range(t)]).sum(axis=0)
        else:
            self.A += np.outer(self.Δv[t-1], self.u[t-1])
            self.B += np.outer(self.u[t-1], self.u[t-1])

        self.prev_t = t
        return self.A @ np.linalg.pinv(self.B)
