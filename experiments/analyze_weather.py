"""Price of weather diversity: FiSER vs FiSER-W (at most one path on satellite links), six OGS.

FiSER rows come from results/journal/spaceground/main_<topology>, FiSER-W rows from
results/journal/spaceground/weather_<topology> (experiments/journal_runs.sh sat_weather). Both use the same
snapshots and parameters; the rate floor S_min is applied to both.

Example:
  .venv/bin/python experiments/analyze_weather.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fiser.report import apply_floor  # noqa: E402

R = ROOT / "results" / "journal" / "spaceground"
S_MIN, N_OGS = 0.01, 6
KEY = ["ti", "s", "d", "F_th"]


def main():
    import argparse
    global R
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(R))
    R = Path(ap.parse_args().root)
    print(f"service floor S_min = {S_MIN}, {N_OGS} OGS; windows = snapshots with a satellite link; runs from {R}\n")
    for topo in ("nsfnet", "cost239", "usnet"):
        wpath = R / f"weather_{topo}" / f"{topo}_raw.csv.gz"
        if not wpath.exists():
            print(f"{topo}: no weather run yet")
            continue
        main_ = apply_floor(pd.read_csv(R / f"main_{topo}" / f"{topo}_raw.csv.gz"), S_MIN)
        wthr = apply_floor(pd.read_csv(wpath), S_MIN)
        f = main_[(main_.method == "FiSER") & (main_.ogs == N_OGS)].set_index(KEY)
        w = wthr[(wthr.method == "FiSER-W") & (wthr.ogs == N_OGS)].set_index(KEY)
        g0 = main_[(main_.method == "FiSER") & (main_.ogs == 0)].groupby("F_th").served.mean()
        assert len(f) == len(w) and f.index.sort_values().equals(w.index.sort_values())
        w = w.loc[f.index]
        assert (w.worst <= f.worst * (1 + 1e-6) + 1e-12).all(), "FiSER-W cannot beat FiSER"
        assert (wthr[wthr.served].sat_paths <= 1).all(), "weather diversity violated"
        same = f.served & ~f.uses_sat  # FiSER's design is weather-diverse, hence optimal for FiSER-W too
        assert (w.served[same]).all() and ((w.worst[same] - f.worst[same]).abs() <= 1e-6 * f.worst[same]).all()
        win = f.n_sat_links > 0
        print(f"=== {topo.upper()}")
        for F in sorted(f.index.get_level_values("F_th").unique()):
            m = f.index.get_level_values("F_th") == F
            fw, ww = f[m & win], w[m & win]
            both_lost = (fw.served & ~ww.served).sum()
            sat_served = fw.served & fw.uses_sat
            rel = (ww.worst[fw.served & ww.served] / fw.worst[fw.served & ww.served])
            print(f"F_th={F:.2f}: ground {100 * g0[F]:.1f}% | windows FiSER {100 * fw.served.mean():.1f}% "
                  f"-> FiSER-W {100 * ww.served.mean():.1f}% (requests lost {both_lost} of {fw.served.sum()}) | "
                  f"24 h {100 * f[m].served.mean():.1f}% -> {100 * w[m].served.mean():.1f}% | "
                  f"gain kept {100 * (ww.served.mean() - g0[F]) / max(fw.served.mean() - g0[F], 1e-12):.0f}% | "
                  f"worst-case S kept on served (geo-mean) {np.exp(np.log(rel).mean()):.3f}, "
                  f"min {rel.min():.3f} | FiSER-W designs using a satellite "
                  f"{100 * (ww.served & ww.uses_sat).sum() / max(ww.served.sum(), 1):.0f}% "
                  f"(FiSER {100 * sat_served.sum() / max(fw.served.sum(), 1):.0f}%)")
        print()


if __name__ == "__main__":
    main()
