"""Helper functions used by notebooks/analysis*.ipynb files."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from utils import savefig


V_MIN, V_MAX = 11.4, 12.6  # 12 ± 5%, units kV
V_MIN2, V_MAX2 = V_MIN**2, V_MAX**2  # units kV^2

# Match Fig8 in Qu and Li (2020)
# - they count the substation as bus 1
# - we count the substation as bus 0
BUSES = (8, 18, 21, 30, 39, 45, 54)  # 0 = substation

TIME_TICKS =  (   0, 2400, 4800,  7200,  9600, 12000, 14400)
TIME_LABELS = ('0h', '4h', '8h', '12h', '16h', '20h', '24h')

Y_MIN, Y_MAX = 11.2, 12.8  # good for plotting V_MIN to V_MAX
YTICKS = (11.4, 11.7, 12, 12.3, 12.6)


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

    Args
    - key: tuple (info_provided, seed), where seed may be None when info_provided == 'known'
    - pkl: saved results read from pickle
    - ax: optional axes to plot a histogram of violations

    Returns
    - num_mistakes: total number of mistakes
    - avg_viol: averge violation magnitude
    - max_viol: maximum violation magnitude
    """
    vs = pkl['vs']
    assert vs.shape == (T, n)

    violates_max = (vs > V_MAX2 + 0.05)  # shape [T, n]
    violates_min = (vs < V_MIN2 - 0.05)  # shape [T, n]
    is_mistake = (violates_max.any(axis=1) | violates_min.any(axis=1))  # shape [T]

    num_mistakes = is_mistake.sum()
    num_bus_step_violations = violates_max.sum() + violates_min.sum()

    all_violations = np.concatenate([
        vs[violates_max] - V_MAX2,
        V_MIN2 - vs[violates_min]
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


def plot_voltages(
    vpars: np.ndarray, buses: Sequence[int] = BUSES,
    ylim: tuple[float, float] | None = (Y_MIN, Y_MAX),
    yticks: tuple[float, ...] | None = YTICKS,
    plots_dir: str = '', filename: str = '', legend_filename: str = ''
) -> None:
    """Plots voltage curves for selected buses.

    Args
    - vpars: shape [T, n], squared voltage magnitdues (units kV^2)
    - buses: which buses to plot, 0 = substation
    - ylim: (ymin, ymax), set to None to use matplotlib defaults
    - yticks: sequence of tick locations, set to None to use matplotlib defaults
    - plots_dir: where to save plots
    - filename: filename (without extension) for saving plot
    - legend_filename: filename (without extension) for saving legend plot
    """
    T = vpars.shape[0]
    ts = range(T)

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100, tight_layout=True)
    for i in np.array(buses) - 1:
        ax.plot(ts, np.sqrt(vpars[:, i]), label=f'{i+1}')

    ax.axhline(V_MIN, ls='--', color='black')
    ax.axhline(V_MAX, ls='--', color='black')
    ax.set(ylabel='Voltage (kV)')
    if ylim is not None:
        ax.set(ylim=ylim)
    if yticks is not None:
        ax.set(yticks=yticks)
    ax.set(xlabel='time $t$', xlim=(-50, T),
           xticks=TIME_TICKS, xticklabels=TIME_LABELS)

    if filename != '':
        assert plots_dir != ''
        savefig(fig, plots_dir, filename=f'{filename}.pdf')
        savefig(fig, plots_dir, filename=f'{filename}.png')
        savefig(fig, plots_dir, filename=f'{filename}.svg')

    if legend_filename != '':
        assert plots_dir != ''
        leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='bus')
        fig.canvas.draw()
        bbox = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        savefig(fig, plots_dir, filename=f'{legend_filename}.pdf', bbox_inches=bbox)
        savefig(fig, plots_dir, filename=f'{legend_filename}.png', bbox_inches=bbox)
        savefig(fig, plots_dir, filename=f'{legend_filename}.svg', bbox_inches=bbox)


def plot_error_and_etahat(
    pkls_dict: dict[str, dict[str, Any]], plots_dir: str, filename: str,
    legend_loc: str | None, etamax: float | None = None
) -> None:
    """Plots model error on left axis, etahat on right axis.

    Args
    - pkls_dict: name (str) => results dict loaded from pickle (dict[str, Any])
    - plots_dir: where to save plots
    - filename: filename (without extension) for saving plot
    - legend_loc: one of [None, 'top', 'separate']
    - etamax: optional upper limit for etahat axis
    """
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100, tight_layout=True)
    axr = ax.twinx()
    axr.spines['right'].set_visible(True)

    for name, pkl in pkls_dict.items():
        print(name)
        dists = pkl['dists']
        T = pkl['vs'].shape[0]
        ax.step(list(dists['t']) + [T], list(dists['X_true']) + [dists['X_true'][-1]],
                where='post', label=name)
        ax.scatter(0, dists['X_true'][0])
        if 'η' in dists:
            axr.step([0] + list(dists['t']) + [T],
                     [0] + list(dists['η']) + [dists['η'][-1]], ':',
                     where='post')
        else:
            axr.plot([0, T], [8.65, 8.65], ':')

    ax.set_ylabel(r'$||\hat{X}_t - X^\star||_\bigtriangleup$')
    axr.set_ylabel(r'$\hat\eta$')
    ax.set(xticks=TIME_TICKS, xticklabels=TIME_LABELS)
    ax.set(xlabel='time $t$', xlim=(-50, T))

    if etamax is not None:
        axr.set_ylim(-0.4, etamax)

    if legend_loc == 'top':
        ax.legend(ncols=2, bbox_to_anchor=(0, 1), loc='lower left')

    savefig(fig, plots_dir, filename=f'{filename}.pdf')
    savefig(fig, plots_dir, filename=f'{filename}.png')
    savefig(fig, plots_dir, filename=f'{filename}.svg')

    if legend_loc == 'separate':
        axr.set_ylabel('')
        axr.set_yticklabels([])
        leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.canvas.draw()
        bbox = leg.get_tightbbox().transformed(fig.dpi_scale_trans.inverted())
        savefig(fig, plots_dir, f'{filename}_legend.pdf', bbox_inches=bbox)
        savefig(fig, plots_dir, f'{filename}_legend.png', bbox_inches=bbox)
        savefig(fig, plots_dir, f'{filename}_legend.svg', bbox_inches=bbox)


def plot_fill(
    ax: plt.Axes, values: np.ndarray, color: int, label: str, alpha: bool = False
) -> None:
    """
    Args
    - ax: axes
    - values: shape [num_runs, T]
    - color: int, index into tab20 colors
        0 = blue, 2 = orange, 4 = green, 7 = purple
    - label: line label
    - alpha: whether to use translucent shading
    """
    num_runs, T = values.shape
    ts = range(T)
    dark = plt.cm.tab20.colors[color]
    light = plt.cm.tab20.colors[color + 1]

    if num_runs == 1:
        ax.plot(ts, values[0], color=dark, lw=0.5, label=label)
    else:
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        ax.plot(ts, mean, color=dark, lw=0.5, label=label)
        if alpha:
            ax.fill_between(ts, mean-std, mean+std, color=light, alpha=0.5)
        else:
            ax.fill_between(ts, mean-std, mean+std, color=light)


def plot_bus(
    pkls_by_label: dict[str, Sequence[dict[str, Any]]], bus: int, plots_dir: str = '',
    legend: bool = False, filename_base: str = ''
) -> None:
    """
    Args:
    - pkls_by_label: label (str) -> list of results dicts
    - bus: int, where bus 0 = substation
    - plots_dir: where to save plots
    - legend: whether to include legend
    - filename_base: base filename for plots
    """
    fig, ax = plt.subplots(figsize=(4, 3), tight_layout=True)
    for c, (label, pkls) in enumerate(pkls_by_label.items()):
        # num_runs = len(pkls)
        vs = np.stack([
            np.sqrt(data['vs'][:, bus - 1])
            for data in pkls
        ], axis=0)
        # vs = np.zeros([num_runs, T])
        # for i, data in enumerate(pkls):
        #     vs[i] = np.sqrt(data['vs'][:, bus - 1])
        plot_fill(ax, vs, color=c*2, label=label, alpha=True)

    T = vs.shape[1]

    ax.axhline(11.4, ls='--', color='black')
    ax.axhline(12.6, ls='--', color='black')
    ax.set(ylabel='Voltage (kV)', ylim=(Y_MIN, Y_MAX), yticks=YTICKS)
    ax.set(xlabel='time $t$', xlim=(-50, T),
           xticks=TIME_TICKS, xticklabels=TIME_LABELS)

    if filename_base != '':
        assert plots_dir != ''

        savefig(fig, plots_dir, filename=f'{filename_base}_bus{bus}.pdf')
        savefig(fig, plots_dir, filename=f'{filename_base}_bus{bus}.png')
        savefig(fig, plots_dir, filename=f'{filename_base}_bus{bus}.svg')

        if legend:
            leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            fig.canvas.draw()
            bbox = leg.get_tightbbox().transformed(fig.dpi_scale_trans.inverted())
            savefig(fig, plots_dir, f'{filename_base}_legend.pdf', bbox_inches=bbox)
            savefig(fig, plots_dir, f'{filename_base}_legend.png', bbox_inches=bbox)
            savefig(fig, plots_dir, f'{filename_base}_legend.svg', bbox_inches=bbox)
