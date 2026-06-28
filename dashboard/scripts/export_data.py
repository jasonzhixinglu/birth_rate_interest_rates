"""
Export model output to JSON for the interactive dashboard.

For each country we:
  1. Calibrate b_high/b_low, solve the pre/post BGP interest rates.
  2. Simulate cohort sizes and solve the OLG transition for r_path.
  3. Re-derive the converged wage path and re-solve every birth cohort's
     household problem (this mirrors the final iteration of solve_transition,
     which itself only returns r_path).
  4. For each snapshot year, record the age distribution (cohort size by age)
     and the per-capita asset profile by age.

Output (written into dashboard/public/data/):
  manifest.json                 list of countries + global meta
  countries/<slug>.json         r-path + per-year snapshots for one country

Run from anywhere:  python dashboard/scripts/export_data.py
"""

import json
import sys
from pathlib import Path

import numpy as np

# repo root = two levels up from this file (dashboard/scripts/export_data.py)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.demographics import bgp_growth_rate, simulate_cohort_sizes
from src.bgp import bgp_interest_rate
from src.olg import solve_transition
from src.household import solve_household
from src.firm import k_from_r, wage_from_k

OUT_DIR = ROOT / "dashboard" / "public" / "data"

# Shared structural parameters (Germany baseline; same across countries).
BASE = dict(alpha=0.33, sigma=0.4, delta=0.05, beta=0.946,
            theta=2.0, gamma=0.015, J=80, chi=20, psi=60)
F_LOW, F_HIGH = 20, 30

# Calendar window shown in the dashboard.
SNAP_START, SNAP_END = 1950, 2050

# Country configs (mirrors notebooks/03_cross_country.ipynb).
COUNTRIES = [
    dict(slug="germany", name="Germany", tfr_high=2.5, tfr_low=1.4, shock_year=1970, color="#6366f1"),
    dict(slug="usa",     name="USA",     tfr_high=2.5, tfr_low=1.8, shock_year=1970, color="#22d3ee"),
    dict(slug="japan",   name="Japan",   tfr_high=2.0, tfr_low=1.4, shock_year=1975, color="#34d399"),
    dict(slug="china",   name="China",   tfr_high=3.0, tfr_low=1.6, shock_year=1980, color="#fbbf24"),
]


def round_list(arr, dp):
    return [round(float(x), dp) for x in arr]


def solve_country(cfg):
    p = dict(BASE)
    J, chi, psi = p["J"], p["chi"], p["psi"]

    b_high = cfg["tfr_high"] / 20.0
    b_low = cfg["tfr_low"] / 20.0
    shock = cfg["shock_year"]

    g_high = bgp_growth_rate(b_high, J=J, F_low=F_LOW, F_high=F_HIGH)
    g_low = bgp_growth_rate(b_low, J=J, F_low=F_LOW, F_high=F_HIGH)

    r_init = bgp_interest_rate(g_high, b_high, **p)
    r_terminal = bgp_interest_rate(g_low, b_low, **p)

    dem = simulate_cohort_sizes(b_high, b_low, shock_year=shock,
                                base_year=1900, end_year=2200,
                                J=J, F_low=F_LOW, F_high=F_HIGH)
    cohort_sizes = np.asarray(dem["cohort_sizes"], dtype=float)
    years = np.asarray(dem["years"], dtype=int)

    res = solve_transition(cohort_sizes, years, r_init=r_init, r_terminal=r_terminal,
                           phi=0.05, max_iter=2000, tol=1e-6, **p)
    r_path = np.asarray(res["r_path"], dtype=float)

    # ── Re-derive cohort asset profiles at the converged r_path ──
    # (mirrors the inner loop of solve_transition's final iteration)
    beta_eff = p["beta"] * (1.0 + p["gamma"]) ** (1.0 - p["theta"]) if p["gamma"] else p["beta"]
    y0 = int(years[0])
    T = len(years)

    k_path = np.array([k_from_r(r, p["alpha"], p["sigma"], p["delta"]) for r in r_path])
    w_path = np.array([wage_from_k(k, p["alpha"], p["sigma"]) for k in k_path])

    g_pre = float(cohort_sizes[1] / cohort_sizes[0]) - 1.0

    def get_cohort_size(by):
        if y0 <= by <= int(years[-1]):
            return float(cohort_sizes[by - y0])
        if by < y0:
            return float(cohort_sizes[0] * (1.0 + g_pre) ** (by - y0))
        return float(cohort_sizes[-1])

    cohort_assets = {}
    for by in range(y0 - J + 1, int(years[-1]) + 1):
        yr_range = np.arange(by, by + J)
        clipped = np.clip(yr_range - y0, 0, T - 1)
        sol = solve_household(r_path[clipped], w_path[clipped],
                              chi=chi, psi=psi, J=J, beta=beta_eff, theta=p["theta"])
        cohort_assets[by] = sol["assets"]

    # ── Build per-year snapshots ──
    snap_years = list(range(SNAP_START, SNAP_END + 1))
    ages = list(range(J))
    snapshots = {}
    pop_max = asset_pc_max = wealth_max = 0.0

    for t in snap_years:
        pop = np.zeros(J)
        asset_pc = np.zeros(J)
        for age in ages:
            by = t - age
            pop[age] = get_cohort_size(by)
            a = cohort_assets.get(by)
            asset_pc[age] = float(a[age]) if a is not None else 0.0
        wealth = pop * asset_pc
        pop_max = max(pop_max, float(pop.max()))
        asset_pc_max = max(asset_pc_max, float(asset_pc.max()))
        wealth_max = max(wealth_max, float(wealth.max()))
        snapshots[str(t)] = {
            "pop": round_list(pop, 4),
            "assetPc": round_list(asset_pc, 5),
        }

    # r-path over the shown window, in percent
    mask = (years >= SNAP_START) & (years <= SNAP_END)
    r_series = [
        {"year": int(y), "r": round(float(r) * 100.0, 4)}
        for y, r in zip(years[mask], r_path[mask])
    ]

    trough_idx = int(np.argmin(r_path))

    meta = {
        "slug": cfg["slug"],
        "name": cfg["name"],
        "color": cfg["color"],
        "tfrHigh": cfg["tfr_high"],
        "tfrLow": cfg["tfr_low"],
        "shockYear": shock,
        "rInit": round(float(r_init) * 100, 3),
        "rTerminal": round(float(r_terminal) * 100, 3),
        "gHigh": round(float(g_high) * 100, 3),
        "gLow": round(float(g_low) * 100, 3),
        "trough": {"year": int(years[trough_idx]), "value": round(float(r_path[trough_idx]) * 100, 3)},
        "J": J, "chi": chi, "psi": psi,
        "popMax": round(pop_max, 4),
        "assetPcMax": round(asset_pc_max, 5),
        "wealthMax": round(wealth_max, 5),
        "snapStart": SNAP_START, "snapEnd": SNAP_END,
    }

    return {"meta": meta, "ages": ages, "rPath": r_series, "snapshots": snapshots}


def main():
    (OUT_DIR / "countries").mkdir(parents=True, exist_ok=True)
    manifest = {"countries": [], "params": BASE}

    for cfg in COUNTRIES:
        print(f"Solving {cfg['name']} ...", flush=True)
        data = solve_country(cfg)
        out = OUT_DIR / "countries" / f"{cfg['slug']}.json"
        out.write_text(json.dumps(data, separators=(",", ":")))
        m = data["meta"]
        manifest["countries"].append({
            "slug": m["slug"], "name": m["name"], "color": m["color"],
            "rInit": m["rInit"], "rTerminal": m["rTerminal"],
            "trough": m["trough"], "shockYear": m["shockYear"],
        })
        size_kb = out.stat().st_size / 1024
        print(f"  -> {out.relative_to(ROOT)}  ({size_kb:.0f} KB)  "
              f"r_init={m['rInit']}%  r_term={m['rTerminal']}%  "
              f"trough={m['trough']['value']}% ({m['trough']['year']})")

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest with {len(manifest['countries'])} countries -> "
          f"{(OUT_DIR / 'manifest.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
