#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio
import scipy.stats as stats
import matplotlib.pyplot as plt


# -----------------------------
#   Basic helper
# -----------------------------

def load_matlab(filename: str):
    return sio.loadmat(filename)


# -----------------------------
#   Color definitions
# -----------------------------
# 直方图：4 张图分别 4 种颜色
HIST_COLORS = {
    "da":      "#1f77b4",  # blue
    "gaba":    "#2ca02c",  # green
    "td":      "#ff7f0e",  # orange
    "dist_td": "#d62728",  # red
}
HIST_LINE_COLOR = "#333333"

# 时间 course 用的 3 条 cue 颜色
CUE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # blue, orange, green


# -----------------------------
#   Figure 3ab: histograms
# -----------------------------

def plot_hist(data, bar_color, line_color, ax=None):
    centers = data['hcenters'].flatten()
    height = data['ckhist'].flatten()
    width = 1.0

    if ax is None:
        ax = plt.gca()

    # 柱子
    ax.bar(
        centers - width / 2.0,
        height,
        width=width,
        color=bar_color,
        edgecolor='black',
        linewidth=1.0,
    )

    # 叠加 t 分布参考曲线
    xs = np.linspace(-13.0, 13.0, 2000)
    ref = stats.t.pdf(xs, 20)
    ref /= ref.max()
    ax.plot(xs, ref, color=line_color, linewidth=2.0, alpha=1.0)

    ax.set_xlim([-13.0, 13.0])
    ax.set_xlabel("t-Statistic", fontsize=14)
    ax.set_ylabel("Relative frequency", fontsize=14)
    ax.tick_params(labelsize=12)


def make_hist_figures_separate():
    fig3_da_hist = load_matlab("DopamineTHist.mat")
    fig3_ga_hist = load_matlab("GABATHist.mat")
    fig3_simTD_hist = load_matlab("SimulationClassicalHist.mat")
    fig3_simDTD_hist = load_matlab("SimulationDistHist.mat")

    # 1. DA response
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plot_hist(fig3_da_hist, bar_color=HIST_COLORS["da"], line_color=HIST_LINE_COLOR, ax=ax)
    ax.set_title("DA response", fontsize=16)
    fig.tight_layout()
    fig.savefig("figure3_da_hist.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 2. GABAergic response
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plot_hist(fig3_ga_hist, bar_color=HIST_COLORS["gaba"], line_color=HIST_LINE_COLOR, ax=ax)
    ax.set_title("GABAergic response", fontsize=16)
    fig.tight_layout()
    fig.savefig("figure3_gaba_hist.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 3. Classical TD
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plot_hist(fig3_simTD_hist, bar_color=HIST_COLORS["td"], line_color=HIST_LINE_COLOR, ax=ax)
    ax.set_title("Classical TD", fontsize=16)
    fig.tight_layout()
    fig.savefig("figure3_classicalTD_hist.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 4. Distributional TD
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plot_hist(fig3_simDTD_hist, bar_color=HIST_COLORS["dist_td"], line_color=HIST_LINE_COLOR, ax=ax)
    ax.set_title("Distributional TD", fontsize=16)
    fig.tight_layout()
    fig.savefig("figure3_distTD_hist.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# -----------------------------
#   Timecourse helper
# -----------------------------

def plot_varexp_cell(resp, sem, times, cue='reward', window=(0.0, 200.0)):
    fig = plt.figure()
    ax = plt.gca()

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for c in range(resp.shape[0]):
        ax.fill_between(
            times,
            resp[c] - sem[c],
            resp[c] + sem[c],
            alpha=0.2,
            color=CUE_COLORS[c],
            zorder=3,
        )
        ax.plot(times, resp[c], color=CUE_COLORS[c], zorder=4)

    ylm = ax.get_ylim()
    ax.fill_between(
        [window[0], window[1]],
        ylm[0],
        ylm[1],
        color='grey',
        alpha=0.05,
        zorder=1,
    )
    ax.axvline(0.0, color='k', linestyle='--', zorder=1)

    ax.set_xlabel(f"Time from {cue} onset (ms)", fontsize=14)
    ax.set_ylabel(r"$\Delta$ Firing Rate (Hz)", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()

    fig.tight_layout()
    return fig


# -----------------------------
#   Figure 3c: DA timecourses
# -----------------------------

def make_da_timecourses():
    fig3_da_psth = load_matlab("AllPSTHDopamine.mat")

    # 汇总不同 outcome，专注 cue time
    cue_resp = []
    for cues in [(0, 5), (1, 4), (2, 3)]:
        cue_resp.append(
            np.hstack([
                fig3_da_psth['PSTH'][:, cues[0]],
                fig3_da_psth['PSTH'][:, cues[1]],
            ])
        )

    cue_resp = np.array(cue_resp)
    cue_resp = np.transpose(cue_resp, (1, 0, 2, 3))

    start, stop = 500, 1500
    psth = np.nanmean(cue_resp, -2)[..., start:stop]
    num_trials = (1.0 - np.isnan(cue_resp).astype(np.float32)).sum(-2)[..., start:stop]

    psth_std = np.nanstd(cue_resp, -2)[..., start:stop]
    psth_sem = psth_std / np.sqrt(num_trials)
    times = fig3_da_psth['psthTimes'][0, start:stop]


    cells_to_plot = fig3_da_psth['K'][0]

    for cell in cells_to_plot:
        cell_int = int(cell)
        idx = cell_int - 1  # MATLAB -> Python

        fig = plot_varexp_cell(
            psth[idx],
            psth_sem[idx],
            times,
            cue='odor',
            window=(0.0, 400.0),
        )
        fig.suptitle(f"Figure 3c - DA cell {cell_int}", fontsize=14)
        fig.savefig(
            f"figure3c_da_cell{cell_int}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)


# -----------------------------
#   Figure 3d: GABA timecourses
# -----------------------------

def make_gaba_timecourses():
    fig3_ga_psth = load_matlab("AllPSTHGABA.mat")

    ga_cue_resp = []
    for cues in [(0, 5), (1, 4), (2, 3)]:
        ga_cue_resp.append(
            np.hstack([
                fig3_ga_psth['PSTH'][:, cues[0]],
                fig3_ga_psth['PSTH'][:, cues[1]],
            ])
        )

    ga_cue_resp = np.array(ga_cue_resp)
    ga_cue_resp = np.transpose(ga_cue_resp, (1, 0, 2, 3))

    start, stop = 500, 1500
    psth = np.nanmean(ga_cue_resp, -2)[..., start:stop]
    num_trials = (1.0 - np.isnan(ga_cue_resp).astype(np.float32)).sum(-2)[..., start:stop]

    psth_std = np.nanstd(ga_cue_resp, -2)[..., start:stop]
    psth_sem = psth_std / np.sqrt(num_trials)
    times = fig3_ga_psth['psthTimes'][0, start:stop]

    for cell_int in [26, 27]:
        idx = cell_int - 1

        fig = plot_varexp_cell(
            psth[idx],
            psth_sem[idx],
            times,
            cue="odor",
            window=(0.0, 1500.0),
        )
        fig.suptitle(f"Figure 3d - GABA cell {cell_int}", fontsize=14)
        fig.savefig(
            f"figure3d_gaba_cell{cell_int}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)


# -----------------------------
#   Main
# -----------------------------

def main():
    # 直方图
    make_hist_figures_separate()

    # 如需一起生成时间 course，就把下面两行取消注释
    # make_da_timecourses()
    # make_gaba_timecourses()

    print("Done. Generated Figure 3 histogram panels as separate PNGs.")


if __name__ == "__main__":
    main()
