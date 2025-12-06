#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Figure 2ab and Figure 2d from the notebook (modified version)
======================================================================
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline



ALL_CONDITIONS = ["0.1uL", "0.3uL", "1.2uL", "2.5uL", "5uL", "10uL", "20uL"]
all_juice_amounts = np.array([0.1, 0.3, 1.2, 2.5, 5.0, 10.0, 20.0])
all_juice_probs = np.array(
    [0.06612594, 0.09090909, 0.14847358, 0.15489467, 0.31159175, 0.1509519, 0.07705306]
)

SELECTED_IND = np.array([0, 1, 2, 4, 6], dtype=int)

conditions = [ALL_CONDITIONS[i] for i in SELECTED_IND]
juice_amounts = all_juice_amounts[SELECTED_IND]
juice_probs = all_juice_probs[SELECTED_IND]
juice_probs = juice_probs / juice_probs.sum()  # 重新归一化


reward_cmap = np.array(
    [
        (0.0, 1.0, 1.0, 1.0),   # cyan
        (1.0, 1.0, 0.0, 1.0),   # yellow
        (1.0, 0.0, 1.0, 1.0),   # magenta
        (0.5, 1.0, 0.0, 1.0),   # lime-ish
        (1.0, 0.5, 0.0, 1.0),   # orange
    ]
)

raster_cmap = plt.cm.bone_r

# utility & response functions (from the notebook)
fmax = 10.0
sigma = 200.0


def utility_func(r):
    """
    Utility function used in the simulation:
        u(r) = fmax * sign(r) * |r|^0.5 / (|r|^0.5 + sigma^0.5)
    """
    r = np.asarray(r, dtype=float)
    return (fmax * np.sign(r) * np.abs(r) ** 0.5) / (
        np.abs(r) ** 0.5 + sigma ** 0.5
    )


def response_func(r):
    """Linear response function."""
    return r


# --------------------------------------------
#  Figure 2b helper: zero-crossing plot
# --------------------------------------------

def plot_zero_crossings(zero_crossings, responses, smooth: float = 10.0):
    zero_crossings = np.asarray(zero_crossings)
    responses = np.asarray(responses)

    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # --- 黑色背景 ---
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # 按 zero-crossing 排序 cell
    ind = np.argsort(zero_crossings)
    juice_response = responses[ind]

    xk = np.arange(len(ind))
    xs = np.linspace(0.0, len(ind) - 1.0, 400)

    # 先画每个 cell 的 min/max 范围
    for cell in range(len(juice_response)):
        ymin = juice_response[cell].min()
        ymax = juice_response[cell].max()
        ax.plot(
            np.ones(2) * cell,
            [ymin, ymax],
            linewidth=1,
            linestyle=":",
            color="0.5",
            zorder=1,
        )

    # 逐条 reward 曲线画散点 + 平滑曲线
    n_rewards = juice_response.shape[1]
    for reward_ind in range(n_rewards):
        color_idx = reward_ind  # 0..4
        # 散点：x=cell index, y=response
        ax.scatter(
            np.arange(len(juice_response)),
            juice_response[:, reward_ind],
            c=[reward_cmap[color_idx]],
            linewidth=1,
            edgecolor="w",
            zorder=4,
            s=40,
            label=conditions[color_idx],
        )

        # Down weight endpoints for the interpolating spline
        weights = np.ones(len(ind))
        weights[0] = 0.5
        weights[-1] = 0.5

        cs = UnivariateSpline(
            xk,
            juice_response[:, reward_ind],
            w=weights,
            s=smooth,
        )

        ax.plot(
            xs,
            cs(xs),
            c=reward_cmap[color_idx],
            zorder=3,
            linewidth=2,
            alpha=1.0,
        )

    # y=0 的横线
    ax.axhline(
        0.0,
        linestyle="-",
        color="w",
        zorder=2,
        linewidth=2.0,
    )

    # cell 轴范围
    ax.set_xlim([-1, len(juice_response)])

    # 坐标轴刻度 & 颜色
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.tick_left()

    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")

    ax.set_xlabel("Cell (sorted by zero-crossing)", fontsize=14, color="black")
    ax.set_ylabel(r"$\Delta$ Firing Rate (variance normalized)", fontsize=14, color="black")

    legend = ax.legend(facecolor="white", edgecolor="black", fontsize=9)
    for text in legend.get_texts():
        text.set_color("black")

    return fig


# --------------------------------------------
#  Figure 2b: data + simulations
# --------------------------------------------

def load_matlab(filename: str):
    return sio.loadmat(filename)


def generate_figure2b_data_panel(fig2b_mat_path="Figure2b.mat"):

    fig2b = load_matlab(fig2b_mat_path)

    zero_crossings = fig2b["zeroCrossings"].flatten()
    responses_all = np.nanmean(fig2b["dataExpectedNorm"], axis=-1)
    responses_5 = responses_all[:, SELECTED_IND]

    fig = plot_zero_crossings(zero_crossings, responses_5)
    fig.suptitle("Figure 2b - Zero crossings (data, 5 rewards)", fontsize=16, color="white")
    fig.tight_layout()
    fig.savefig(
        "figure2b_zero_crossings_data_5rewards.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def generate_figure2a_simulation_panels():

    # fixed seed for reproducibility
    np.random.seed(0)

    num_cells = 151
    num_steps = 25000
    trials = 10
    base_lrate = 0.02

    # ----- Classical TD -----
    values = np.zeros((trials, num_cells))
    alpha = np.random.random((trials, num_cells))

    for trial in range(trials):
        for step in range(num_steps):
            reward = np.random.choice(juice_amounts, p=juice_probs)
            delta = utility_func(reward) - values[trial]
            values[trial] += alpha[trial] * response_func(delta) * base_lrate

    # sort by value
    ind = np.argsort(values)
    values = np.array([v[i] for v, i in zip(values, ind)])
    alpha = np.array([a[i] for a, i in zip(alpha, ind)])

    # compute responses across 5 reward conditions
    responses = response_func(
        alpha[:, :, None]
        * (utility_func(juice_amounts)[None, None, :] - values[:, :, None])
    )
    ind = np.argsort(values.mean(0))
    responses = responses.mean(0)
    responses = responses[ind]
    responses /= responses.std(1, ddof=1, keepdims=True)

    fig_td = plot_zero_crossings(values.mean(0), responses)
    fig_td.suptitle("Figure 2a - Classical TD simulation (5 rewards)", fontsize=16, color="white")
    fig_td.tight_layout()
    fig_td.savefig(
        "figure2a_classicalTD_5rewards.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=fig_td.get_facecolor(),
    )
    plt.close(fig_td)

    # ----- Distributional TD -----
    distribution = np.zeros((trials, num_cells))

    alpha_pos = np.random.random(size=(trials, num_cells))
    alpha_neg = np.random.random(size=(trials, num_cells))
    tau = alpha_pos / (alpha_pos + alpha_neg)

    for trial in range(trials):
        for step in range(num_steps):
            reward = np.random.choice(juice_amounts, p=juice_probs)
            delta = utility_func(reward) - distribution[trial]

            valence = np.array(delta <= 0.0, dtype=np.float32)
            distribution[trial] += (
                (valence * alpha_neg[trial] + (1.0 - valence) * alpha_pos[trial])
                * response_func(delta)
                * base_lrate
            )

        ind = np.argsort(tau[trial])
        tau[trial] = tau[trial][ind]
        alpha_pos[trial] = alpha_pos[trial][ind]
        alpha_neg[trial] = alpha_neg[trial][ind]
        distribution[trial] = distribution[trial][ind]

    delta = utility_func(juice_amounts)[None, None, :] - distribution[:, :, None]
    valence = np.array(delta <= 0.0, dtype=np.float32)

    lrfloor = lambda x: 0.2 + x

    responses = response_func(delta) * (
        valence * lrfloor(alpha_neg)[:, :, None]
        + (1.0 - valence) * lrfloor(alpha_pos)[:, :, None]
    )
    responses = responses.mean(0)
    ind = np.argsort(distribution.mean(0))
    responses = responses[ind]
    responses /= responses.std(1, ddof=1, keepdims=True)

    fig_dtd = plot_zero_crossings(distribution.mean(0), responses)
    fig_dtd.suptitle(
        "Figure 2a - Distributional TD simulation (5 rewards)", fontsize=16, color="white"
    )
    fig_dtd.tight_layout()
    fig_dtd.savefig(
        "figure2a_distributionalTD_5rewards.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=fig_dtd.get_facecolor(),
    )
    plt.close(fig_dtd)


# -----------------------------
#  Main entry point
# -----------------------------

def main():
    # Figure 2b: data
    generate_figure2b_data_panel("Figure2b.mat")

    # Figure 2a: simulations (classical TD + distributional TD)
    generate_figure2a_simulation_panels()

    print("Done. Generated PNG files for Figure 2ab (5 rewards) and Figure 2d data.")


if __name__ == "__main__":
    main()
