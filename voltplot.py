from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import IPython.display


TIME_TICKS =  (      0,    2400,    4800,    7200,    9600,   12000,   14400)
TIME_LABELS = ('00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00')


class VoltPlotly:

    def __init__(self, v_lims: tuple[float, float],
                 q_lims: tuple[float, float], widget: bool | None = None):
        """
        Args
        - v_lims: tuple of float (v_min, v_max), units kV (not squared)
        - q_lims: tuple of float (q_min, q_max), units MVar
        - widget: optional bool, whether using `%matplotlib widget`
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        self.is_showing = False

        # Recreate Fig8 in Qu and Li (2020)
        # - they count the substation as bus 1
        # - we count the substation as bus -1
        self.index = [9, 19, 22, 31, 40, 46, 55]

        q_min, q_max = q_lims
        v_min, v_max = v_lims

        fig = make_subplots(rows=2, cols=2, shared_xaxes=True, subplot_titles=[
            'Reactive power injection', 'Convergence of CMC',
            'Voltage Profile', 'Voltage Profile Without Controller',
        ])
        fig = go.FigureWidget(fig)

        fig.add_hline(row=1, col=1, y=q_min, line_dash='dash')
        fig.add_hline(row=1, col=1, y=q_max, line_dash='dash')
        fig.update_yaxes(row=1, col=1, title_text='Reactive Power (MVar)')

        fig.update_yaxes(row=1, col=2, title_text=r'$||\hat{X} - X||_{\Delta}$')

        for c in [1, 2]:
            fig.add_hline(row=2, col=c, y=v_min, line_dash='dash')
            fig.add_hline(row=2, col=c, y=v_max, line_dash='dash')
            fig.update_yaxes(row=2, col=c, title_text='Voltage (kV)')
            fig.update_xaxes(
                row=2, col=c, title_text='time (hh:mm)', tickmode='array',
                tickvals=TIME_TICKS, ticktext=TIME_LABELS)

        # create empty plots, placeholders
        n_buses = len(self.index)
        for (r, c) in [(1, 1), (2, 1), (2, 2)]:
            for i in np.asarray(self.index) - 2:
                fig.add_scatter(row=r, col=c, x=[], y=[], mode='lines')
        self.qcs_lines = fig.data[0:n_buses]
        self.vs_lines = fig.data[n_buses:2*n_buses]
        self.vpars_lines = fig.data[2*n_buses:3*n_buses]

        fig.add_scatter(row=1, col=2, x=[], y=[], mode='lines', line_shape='vh')
        self.dist_line = fig.data[-1]

        self.fig = fig

    def update(self, qcs: np.ndarray, vs: np.ndarray, vpars: np.ndarray,
               dists: tuple[list, list]) -> None:
        """
        Args
        - qcs: np.array, shape [n, T]
        - vs: np.array, shape [n, T]
        - vpars: np.array, shape [n, T]
        - dists: tuple of lists (ts, dists_true)
            - ts: list of int, time steps at which the model was updated
            - dists_true: list of float, ||X̂-X||_△ after each model update
        """
        ts = list(range(qcs.shape[1]))
        # with self.fig.batch_update():
        for l, i in enumerate(np.asarray(self.index) - 2):
            self.qcs_lines[l].x = ts
            self.qcs_lines[l].y = qcs[i]
            self.vs_lines[l].x = ts
            self.vs_lines[l].y = vs[i]
            self.vpars_lines[l].x = ts
            self.vpars_lines[l].y = vpars[i]
        self.dist_line.x = dists[0]
        self.dist_line.y = dists[1]

    def show(self, clear_display: bool = False) -> None:
        if clear_display:
            IPython.display.clear_output()
        if not self.is_showing:
            self.fig.show()
            self.is_showing = True


class VoltPlot:

    def __init__(self, v_lims: tuple[float, float],
                 q_lims: tuple[float, float], widget: bool | None = None):
        """
        Args
        - v_lims: tuple of float (v_min, v_max), units kV (not squared)
        - q_lims: tuple of float (q_min, q_max), units MVar
        - widget: optional bool, whether using `%matplotlib widget`
        """
        # Recreate Fig8 in Qu and Li (2020)
        # - they count the substation as bus 1
        # - we count the substation as bus -1
        self.index = [9, 19, 22, 31, 40, 46, 55]

        if widget is None:
            widget = ('nbagg' in plt.get_backend())
        self.widget = widget
        tqdm.write(f'widget? {self.widget}')

        fig, axs = plt.subplots(1, 4, figsize=(16, 4), dpi=60, tight_layout=True)
        self.fig = fig
        self.axs = axs

        q_min, q_max = q_lims
        v_min, v_max = v_lims

        ax = axs[0]
        ax.axhline(q_min, ls='--')
        ax.axhline(q_max, ls='--')
        ax.set_ylabel('Reactive Power (MVar)')
        ax.set_title('Reactive power injection')

        ax = axs[1]
        ax.axhline(v_min, ls='--')
        ax.axhline(v_max, ls='--')
        ax.set(ylabel='Voltage (kV)')
        ax.set_title('Voltage Profile')

        ax = axs[2]
        ax.axhline(v_min, ls='--')
        ax.axhline(v_max, ls='--')
        ax.set_ylabel('Voltage (kV)')
        ax.set_title('Voltage Profile, no Controller')

        ax = axs[3]
        ax.set_ylabel(r'$||\hat{X} - X||_{\Delta}$')
        ax.set_title('Model Chasing Convergence')

        for ax in axs:
            ax.set_xticks(TIME_TICKS)
            ax.set_xticklabels(TIME_LABELS)

        # create empty plots, placeholders
        self.qcs_lines = []
        self.vs_lines = []
        self.vpars_lines = []
        self.dist_line = axs[3].step([], [], where='post')[0]  # step-function
        for i in np.asarray(self.index) - 2:
            qcs_line, = axs[0].plot([], [], label=f'bus {i+2}')
            vs_line, = axs[1].plot([], [])
            vpars_line, = axs[2].plot([], [])

            self.qcs_lines.append(qcs_line)
            self.vs_lines.append(vs_line)
            self.vpars_lines.append(vpars_line)

        axs[0].legend()
        if widget:
            plt.show()
        else:
            pass
            # plt.close()  # so that the plot doesn't show

    def update(self, qcs: np.ndarray, vs: np.ndarray, vpars: np.ndarray,
               dists: tuple[list, list]) -> None:
        """
        Args
        - qcs: np.array, shape [T, n]
        - vs: np.array, shape [T, n]
        - vpars: np.array, shape [T, n]
        - dists: tuple of lists (ts, dists_true)
            - ts: list of int, time steps at which the model was updated
            - dists_true: list of float, ||X̂-X||_△ after each model update
        """
        ts = range(qcs.shape[0])
        for l, i in enumerate(np.asarray(self.index) - 2):
            self.qcs_lines[l].set_data(ts, qcs[:, i])
            self.vs_lines[l].set_data(ts, vs[:, i])
            self.vpars_lines[l].set_data(ts, vpars[:, i])

        # extend out self.dist_line to match other plots
        self.dist_line.set_data(dists[0] + [ts[-1]], dists[1] + dists[1][-1])

        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()

    def show(self, clear_display: bool = False) -> None:
        if clear_display:
            IPython.display.clear_output()
        if self.widget:
            self.fig.canvas.draw()
        else:
            IPython.display.display(self.fig)
