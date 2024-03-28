"""Helper functions used by notebooks/analysis*.ipynb files."""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


V_MIN, V_MAX = (11.4**2, 12.6**2)  # +/-5%, units kV^2


def calculate_violations(
    key: tuple[str, int | None], pkl: dict[str, Any], T: int, n: int,
    ax: plt.Axes | None = None
) -> tuple[int, float, float]:
    """
    Definitions:
    - a *mistake* is a time step t where some bus voltage violates the voltage limits,
        i.e.,  v(t) ∉ [v̲(t), v̅(t)]. v(t) is a vector with the squared-voltages at
        every bus. As long as a single bus violates the voltage limits at time t,
        time step t is considered a mistake.
    - a *bus-timestep violation* refers to a tuple (t,i) where the voltage of bus i
        at time t violates the voltage limits. That is, vᵢ(t) ∉ [v̲ᵢ(t), v̅ᵢ(t)]
    - we don't count any mistakes / violations where the violation is less than 0.05.
        Anecdotally, it seems that CVXPY sometimes allows for solutions with numerical
        errors up to 0.05.

    Prints out the following information:
    - '# updates': number of time steps where the model X̂ or η-hat was updated
    - 'frac mistakes': fraction of time steps where the model made a mistake
    - '# bus-timestep violations': number of bus-timestep violations
    - 'avg viol': among all bus-timestep violations, the average absolute magnitude
        of the violation
    - 'max viol': among all bus-timestep violations, the maximum absolute magnitude
        of the violation

    Args:
        key: tuple (info_provided, seed), where seed may be None when info_provided == 'known'
        pkl: saved results read from pickle
        ax: optional axes to plot a histogram of violations

    Returns:
        num_mistakes: total number of mistakes
        avg_viol: averge violation magnitude
        max_viol: maximum violation magnitude
    """
    vs = pkl['vs']
    assert vs.shape == (T, n)

    violates_max = (vs > V_MAX + 0.05)  # shape [T, n]
    violates_min = (vs < V_MIN - 0.05)  # shape [T, n]
    is_mistake = (violates_max.any(axis=1) | violates_min.any(axis=1))  # shape [T]

    num_mistakes = is_mistake.sum()
    num_bus_step_violations = violates_max.sum() + violates_min.sum()

    all_violations = np.concatenate([
        vs[violates_max] - V_MAX,
        V_MIN - vs[violates_min]
    ])

    if len(all_violations) == 0:
        avg_viol = max_viol = 0
    else:
        avg_viol = np.mean(all_violations)
        max_viol = np.max(all_violations)

    if ax is not None:
        ax.hist(all_violations, bins=np.arange(0, 8, 0.1))
        ax.set(xlabel='abs. violation', ylabel='count', title=str(key), yscale='log')

    num_updates = len(pkl['dists']['t']) - 1

    print(f'key: {key}, # updates: {num_updates}, '
          f'frac mistakes: {num_mistakes}/{T}, '
          f'# bus-timestep violations: {num_bus_step_violations}, '
          f'avg viol: {avg_viol:.3g}, ',
          f'max viol: {max_viol:.3g}')

    return num_mistakes, avg_viol, max_viol
