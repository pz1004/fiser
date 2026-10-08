"""Paper figures (vector PDF, IEEE column widths) from the results CSVs.

Each method keeps one fixed palette slot and marker in every figure, so color
follows the method. The palette is the validated categorical order (light mode,
white page); markers are the secondary channel for grayscale print.

Example:
  .venv/bin/python experiments/plot_figures.py --outdir paper/figs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
METHODS = {  # method -> (label, palette slot color, marker); slots 1-5 of the validated order
    "FiSER": ("FiSER (proposed)", "#2a78d6", "o"),
    "B2": ("B2: B1 + purification", "#eb6834", "s"),
    "B3": ("B3: min-hop + purif.", "#1baf7a", "D"),
    "B1": ("B1: fidelity-unaware", "#eda100", "^"),
    "B5": ("B5: sequential", "#e87ba4", "v"),
}
COL_W, PAGE_W = 3.5, 7.16  # IEEE column / two-column widths [in]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 7.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK, "axes.titlesize": 8,
    "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "grid.linestyle": "-", "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 2,
    "ytick.major.size": 2, "legend.frameon": False, "legend.fontsize": 7, "lines.linewidth": 1.5,
    "lines.markersize": 4, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def _series(ax, g, x, y, order):
    for m in order:
        h = g[g.method == m].sort_values(x)
        if h.empty:
            continue
        label, color, marker = METHODS[m]
        ax.plot(h[x], h[y], color=color, marker=marker, label=label,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3 if m == "FiSER" else 2)


def fig_ground(summary_csv, out_pdf, order=("FiSER", "B5", "B2", "B3", "B1")):
    """Top: share of SD pairs served (protected, fidelity-feasible, above the rate floor).
    Bottom: mean worst-case success relative to FiSER (<= 1 by construction)."""
    tab = pd.read_csv(summary_csv)
    tab["service_pct"] = 100 * tab.served / tab.pairs
    ref = tab[tab.method == "FiSER"].set_index(["topology", "F_th"]).mean_worst
    tab["rel_worst"] = tab.mean_worst / tab.set_index(["topology", "F_th"]).index.map(ref)
    order_t = ["NSFNET", "COST239", "USNET"]
    topos = sorted(dict.fromkeys(tab.topology), key=lambda t: order_t.index(t) if t in order_t else 99)
    fig, axes = plt.subplots(2, len(topos), figsize=(PAGE_W if len(topos) > 1 else COL_W, 2.75),
                             sharex=True, squeeze=False, gridspec_kw=dict(height_ratios=(1.25, 1)))
    fths = sorted(tab.F_th.unique())
    for j, topo in enumerate(topos):
        g = tab[tab.topology == topo]
        _series(axes[0, j], g, "F_th", "service_pct", order)
        _series(axes[1, j], g, "F_th", "rel_worst", order)
        axes[0, j].set_title(f"{topo} ({g.pairs.iloc[0]} SD pairs)")
        axes[0, j].set_ylim(0, 104)
        axes[1, j].set_ylim(0, 1.04)
        axes[1, j].set_xticks(fths, [f"{f:.2f}" for f in fths])
        axes[1, j].set_xlabel(r"Fidelity threshold $F_{\mathrm{th}}$")
    axes[0, 0].set_ylabel("Served pairs [%]")
    axes[1, 0].set_ylabel("Worst-case $S$\n(rel. to FiSER)")
    fig.align_ylabels(axes[:, 0])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(order) if len(topos) > 1 else 2,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=1.2, labelcolor=INK)
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    fig.savefig(out_pdf)
    plt.close(fig)


THRESH = {0.85: "#2a78d6", 0.90: "#104281"}  # validated ordinal ramp (blue 450/650)
NIGHT_WASH = "#f0efec"
S_MIN = 0.01


def _floored(raw):
    import sys
    sys.path.insert(0, str(ROOT))
    from fiser.report import apply_floor
    return apply_floor(raw, S_MIN)


def fig_satellite(ts_dir, topo_name, out_pdf, n_ogs=4, const="144/12", method="FiSER"):
    """Share of SD pairs served over 24 h with n_ogs OGS (thin lines: ground only); night shaded."""
    import json
    import sys

    import numpy as np
    sys.path.insert(0, str(ROOT))
    from fiser import topologies
    from fiser.network import build_network
    from fiser.physics import PhysParams
    from fiser.satellite import sun_elevation
    from fiser.spaceground import SatParams, place_on_map

    rd = Path(ts_dir)
    meta = json.loads((rd / "params.json").read_text())
    step = meta["args"]["step_s"]
    raw = pd.read_csv(rd / f"{topo_name}_raw.csv.gz")
    raw = _floored(raw[(raw.method == method) & (raw.constellation == const)])

    topo = topologies.load(topo_name)
    G = build_network(topo["edges"], PhysParams(**meta["phys"]), max_link_km=meta["args"]["max_link_km"])
    ll = np.array(list(place_on_map(G, topo, tuple(meta["args"]["center"])).values()))
    S = SatParams(**meta["sat"])
    t = np.arange(0, 86400, 60)
    night = sun_elevation(t + S.epoch_unix(), [ll[:, 0].mean()], [ll[:, 1].mean()])[:, 0] < S.sun_limit_deg

    fig, ax = plt.subplots(figsize=(COL_W, 1.5))
    edges = np.flatnonzero(np.diff(np.r_[0, night.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        ax.axvspan(t[a] / 3600, t[min(b, len(t) - 1)] / 3600, color=NIGHT_WASH, lw=0, zorder=0)
    for F_th, color in THRESH.items():
        sg = raw[(raw.ogs == n_ogs) & (raw.F_th == F_th)].groupby("ti").served.mean() * 100
        sg = sg.rolling(max(1, 600 // step), center=True, min_periods=1).mean()  # 10-min moving average
        gr = raw[(raw.ogs == 0) & (raw.F_th == F_th)].served.mean() * 100
        ax.plot(sg.index * step / 3600, sg.values, color=color, lw=1.2, label=f"$F_{{\\rm th}}$={F_th:.2f}")
        ax.axhline(gr, color=color, lw=0.8, zorder=1)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 102)
    ax.set_xticks(range(0, 25, 4))
    ax.set_xlabel("Hours after 12:00 UTC (shaded: night)")
    ax.set_ylabel("Served pairs [%]")
    ax.legend(loc="upper left", ncol=2, fontsize=6.5, handlelength=1.2, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def table_spaceground(main_dirs, out_tex, n_ogs=6, fths=(0.80, 0.85, 0.90), method="FiSER", weather_dirs=None):
    """LaTeX tabular: served share [%] ground-only, 24-h mean, and during night windows (n_ogs OGS);
    with weather_dirs, also FiSER-W (at most one path on satellite links) during night windows."""
    lines = []
    for name, d in main_dirs.items():
        raw = pd.read_csv(Path(d) / f"{name.lower()}_raw.csv.gz")
        raw = _floored(raw[raw.method == method])
        wraw = None
        if weather_dirs:
            wraw = pd.read_csv(Path(weather_dirs[name]) / f"{name.lower()}_raw.csv.gz")
            wraw = _floored(wraw[(wraw.method == "FiSER-W") & (wraw.ogs == n_ogs)])
        for i, F in enumerate(fths):
            g0 = 100 * raw[(raw.ogs == 0) & (raw.F_th == F)].served.mean()
            r = raw[(raw.ogs == n_ogs) & (raw.F_th == F)]
            day = 100 * r.served.mean()
            win = 100 * r[r.n_sat_links > 0].served.mean()
            vals = [g0, day, win]
            if wraw is not None:
                rw = wraw[wraw.F_th == F]
                vals.append(100 * rw[rw.n_sat_links > 0].served.mean())
            if min(vals) >= 99.95:
                continue  # fully served everywhere; noted in the caption
            label = name if not any(l.startswith(name) for l in lines) else ""
            lines.append(f"{label} & {F:.2f} & " + " & ".join(f"{v:.0f}" for v in vals) + " \\\\")
        lines.append("\\addlinespace[2pt]")
    cols, extra = ("lccccc", " & FiSER-W") if weather_dirs else ("lcccc", "")
    head = (f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n & $F_{{\\mathrm{{th}}}}$ & Ground & 24\\,h & Windows{extra} \\\\\n"
            "\\midrule\n")
    Path(out_tex).write_text(head + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")


def table_scaling(scaling_csv, out_tex):
    """LaTeX table: per size, feasible share, MILP and heuristic time per request (mean / max),
    and heuristic optimality over feasible requests."""
    df = pd.read_csv(scaling_csv)
    lines = []
    for n, g in df.groupby("n"):
        feas = g.milp >= S_MIN  # served under the rate floor
        opt = (g.heur >= g.milp * (1 - 1e-6)) & feas
        lines.append(f"{n} & {100 * feas.mean():.0f} & {1e3 * g.t_milp.mean():.0f} / {1e3 * g.t_milp.max():.0f} & "
                     f"{1e3 * g.t_heur.mean():.0f} / {1e3 * g.t_heur.max():.0f} & {100 * opt.sum() / feas.sum():.1f} \\\\")
    head = ("\\begin{tabular}{rcccc}\n\\toprule\n$|V|$ & Feasible [\\%] & MILP [ms] & Heuristic [ms] & "
            "Optimal [\\%] \\\\\n\\midrule\n")
    Path(out_tex).write_text(head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def table_alpha2(summary_csv, out_tex, fths=(0.70, 0.75, 0.80, 0.85, 0.90), order=("FiSER", "B5", "B3", "B2", "B1")):
    """LaTeX tabular: served pairs under two-failure protection (three disjoint paths)."""
    t = pd.read_csv(summary_csv)
    t["F_th"] = t.F_th.round(2)
    lines = []
    for topo in ("NSFNET", "COST239", "USNET"):
        g = t[t.topology == topo]
        n = int(g.pairs.iloc[0])
        for i, F in enumerate(fths):
            row = g[g.F_th == F].set_index("method").served
            label = f"{topo} ({n})" if i == 0 else ""
            lines.append(f"{label} & {F:.2f} & " + " & ".join(str(int(row[m])) for m in order) + " \\\\")
        lines.append("\\addlinespace[2pt]")
    head = ("\\begin{tabular}{lcccccc}\n\\toprule\n & $F_{\\mathrm{th}}$ & " + " & ".join(order) + " \\\\\n\\midrule\n")
    Path(out_tex).write_text(head + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")


def table_coherence(runs, out_tex, F_th=0.80, order=("FiSER", "B5", "B3", "B2", "B1")):
    """LaTeX tabular: served pairs at F_th for memory coherence times T (runs: {label: summary.csv})."""
    lines = []
    for topo in ("NSFNET", "COST239", "USNET"):
        for i, (label, csv) in enumerate(runs.items()):
            t = pd.read_csv(csv)
            row = t[(t.topology == topo) & (t.F_th.round(2) == round(F_th, 2))].set_index("method").served
            name = topo if i == 0 else ""
            lines.append(f"{name} & {label} & " + " & ".join(str(int(row[m])) for m in order) + " \\\\")
        lines.append("\\addlinespace[2pt]")
    head = ("\\begin{tabular}{llccccc}\n\\toprule\n & $T$ & " + " & ".join(order) + " \\\\\n\\midrule\n")
    Path(out_tex).write_text(head + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")


def table_scaling_both(csv1, csv2, out_tex, csv_large=None):
    """LaTeX tabular: per mesh size, single- and two-failure protection: feasible share,
    MILP and heuristic time per request (mean / max) [ms]; optional large meshes (single failure,
    local requests) marked with a dagger."""
    d1, d2 = pd.read_csv(csv1), pd.read_csv(csv2)

    def cell(g):
        feas = g.milp >= S_MIN
        return (f"{100 * feas.mean():.0f} & {1e3 * g.t_milp.mean():.0f}/{1e3 * g.t_milp.max():.0f} & "
                f"{1e3 * g.t_heur.mean():.0f}/{1e3 * g.t_heur.max():.0f}")
    lines = []
    for n in sorted(d1.n.unique()):
        lines.append(f"{n} & {cell(d1[d1.n == n])} & {cell(d2[d2.n == n])} \\\\")
    if csv_large is not None:
        dl = pd.read_csv(csv_large)
        for n in sorted(dl.n.unique()):
            lines.append(f"{n}$^\\dagger$ & {cell(dl[dl.n == n])} & -- & -- & -- \\\\")
    head = ("\\begin{tabular}{rcccccc}\n\\toprule\n & \\multicolumn{3}{c}{One failure ($N=2$)} & "
            "\\multicolumn{3}{c}{Two failures ($N=3$)} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
            "$|V|$ & Feas.\\,[\\%] & MILP & Heur. & Feas.\\,[\\%] & MILP & Heur. \\\\\n\\midrule\n")
    Path(out_tex).write_text(head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground", default=str(ROOT / "results" / "journal" / "ground" / "summary.csv"))
    ap.add_argument("--scaling", default=str(ROOT / "results" / "journal" / "scaling" / "scaling.csv"))
    ap.add_argument("--outdir", default=str(ROOT / "paper" / "figs"))
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if Path(args.ground).exists():
        fig_ground(args.ground, out / "ground.pdf")
        print("wrote", out / "ground.pdf")
    sgd = ROOT / "results" / "journal" / "spaceground"
    if (sgd / "main_nsfnet").exists():
        fig_satellite(sgd / "main_nsfnet", "nsfnet", out / "satellite.pdf", n_ogs=6)
        print("wrote", out / "satellite.pdf")
    mains = {n: sgd / f"main_{n.lower()}" for n in ("NSFNET", "COST239", "USNET")}
    weather = {n: sgd / f"weather_{n.lower()}" for n in mains}
    if not all((d / f"{n.lower()}_raw.csv.gz").exists() for n, d in weather.items()):
        weather = None
    if all((d / f"{n.lower()}_raw.csv.gz").exists() for n, d in mains.items()):
        table_spaceground(mains, out / "spaceground_table.tex", weather_dirs=weather)
        print("wrote", out / "spaceground_table.tex")
    if Path(args.scaling).exists():
        table_scaling(args.scaling, out / "scaling_table.tex")
        print("wrote", out / "scaling_table.tex")
    J = ROOT / "results" / "journal"
    if (J / "ground_alpha2" / "summary.csv").exists():
        table_alpha2(J / "ground_alpha2" / "summary.csv", out / "alpha2_table.tex")
        print("wrote", out / "alpha2_table.tex")
    coh = {"$\\infty$": J / "ground" / "summary.csv", "1\\,s": J / "ground_T1" / "summary.csv",
           "0.1\\,s": J / "ground_T0.1" / "summary.csv", "10\\,ms": J / "ground_T0.01" / "summary.csv"}
    if all(Path(c).exists() for c in coh.values()):
        table_coherence(coh, out / "coherence_table.tex")
        print("wrote", out / "coherence_table.tex")
    if (J / "scaling" / "scaling.csv").exists() and (J / "scaling_alpha2" / "scaling.csv").exists():
        large = J / "scaling_large" / "scaling.csv"
        table_scaling_both(J / "scaling" / "scaling.csv", J / "scaling_alpha2" / "scaling.csv", out / "scaling_table.tex",
                           large if large.exists() else None)
        print("wrote", out / "scaling_table.tex (single and two failures)")


if __name__ == "__main__":
    main()
