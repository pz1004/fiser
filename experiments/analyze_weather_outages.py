"""Protection against cloud outages: FiSER vs FiSER-W on the recorded space-ground designs.

Weather model per snapshot: with probability rho a regional front covers every OGS; otherwise each OGS is
clear independently with probability p_clear. A satellite link (u, v) works iff both OGS are clear; a path
works iff all its satellite links work (fiber links are unaffected by weather); a design keeps service iff
at least one of its paths works. The probability is computed exactly by enumerating the OGS involved.

Reported per topology and F_th (night windows, six OGS): the share of requests served, the mean probability
that a served design keeps service under weather alone, and the expected share of requests with protected
service (served x weather survival).

Example:
  .venv/bin/python experiments/analyze_weather_outages.py --root results/journal/spaceground
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fiser.report import apply_floor  # noqa: E402

S_MIN = 0.01


def sat_links(design):
    """Per path: list of satellite links as OGS pairs."""
    out = []
    for path in str(design).split("|"):
        parts = path.split(":")
        sats = parts[2] if len(parts) > 2 else ""
        out.append([tuple(map(int, h.split("@")[0].split("-"))) for h in sats.split(";") if h])
    return out


def survival(design, p_clear, rho):
    """P(at least one path works | weather), exact over the OGS that the design's satellite links use."""
    paths = sat_links(design)
    if any(not links for links in paths):
        return 1.0  # a fiber-only path is immune to weather
    ogs = sorted({u for links in paths for link in links for u in link})
    prob = 0.0
    for clear in itertools.product((False, True), repeat=len(ogs)):
        state = dict(zip(ogs, clear))
        pr = 1.0
        for c in clear:
            pr *= p_clear if c else 1 - p_clear
        works = any(all(state[u] and state[v] for u, v in links) for links in paths)
        prob += pr * works
    return (1 - rho) * prob  # under a regional front every satellite link fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "results" / "journal" / "spaceground"))
    ap.add_argument("--ogs", type=int, default=6)
    ap.add_argument("--p-clear", type=float, nargs="+", default=[0.5, 0.7])
    ap.add_argument("--rho", type=float, nargs="+", default=[0.0, 0.2, 0.5])
    args = ap.parse_args()
    root = Path(args.root)
    rows = []
    for topo in ("nsfnet", "cost239", "usnet"):
        m = root / f"main_{topo}" / f"{topo}_raw.csv.gz"
        w = root / f"weather_{topo}" / f"{topo}_raw.csv.gz"
        if not (m.exists() and w.exists()):
            print(f"{topo}: runs missing")
            continue
        df = pd.concat([apply_floor(pd.read_csv(m), S_MIN), apply_floor(pd.read_csv(w), S_MIN)])
        df = df[(df.ogs == args.ogs) & (df.n_sat_links > 0) & df.method.isin(["FiSER", "FiSER-W"])]
        for (F, meth), g in df.groupby(["F_th", "method"]):
            served = g[g.served]
            for pc in args.p_clear:
                for rho in args.rho:
                    surv = served.design.apply(lambda d: survival(d, pc, rho))
                    rows.append(dict(topology=topo, F_th=F, method=meth, p_clear=pc, rho=rho,
                                     served=100 * g.served.mean(), survive=100 * surv.mean(),
                                     protected=100 * surv.sum() / len(g),
                                     both_on_sat=100 * (served.sat_paths >= 2).mean()))
    out = pd.DataFrame(rows)
    out["F_th"] = out.F_th.map(lambda v: f"{v:.2f}")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    out.to_csv(root / "weather_outages.csv", index=False)


if __name__ == "__main__":
    main()
