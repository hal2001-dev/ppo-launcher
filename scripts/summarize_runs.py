"""Summarize TensorBoard runs: convergence step (first time 100-ep moving average >= threshold), final mean100, eval.

Usage:
    python scripts/summarize_runs.py runs/                      # every run under runs/
    python scripts/summarize_runs.py runs/_tuning --filter M_zoo
    python scripts/summarize_runs.py runs/ --threshold 200 --window 100
"""

import argparse
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def scalars(ea, tag):
    return [(e.step, e.value) for e in ea.Scalars(tag)] if tag in ea.Tags()["scalars"] else []


def summarize(run_dir, threshold, window):
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    rets = scalars(ea, "charts/episodic_return")
    if not rets:
        return None
    steps = np.array([s for s, _ in rets])
    vals = np.array([v for _, v in rets])
    ma = np.convolve(vals, np.ones(window) / window, mode="valid") if len(vals) >= window else np.array([])
    hit = np.nonzero(ma >= threshold)[0]
    conv = int(steps[hit[0] + window - 1]) if len(hit) else None
    lens = scalars(ea, "charts/episodic_length")
    trunc = np.mean([v >= 1000 for _, v in lens[-window:]]) if lens else float("nan")
    ev = scalars(ea, "eval/mean_return")
    return {
        "run": run_dir.name, "last_step": int(steps[-1]), "conv_step": conv,
        "final_mean100": float(vals[-window:].mean()), "trunc_frac": float(trunc),
        "eval": ev[-1][1] if ev else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", nargs="?", default="runs")
    p.add_argument("--filter", default="")
    p.add_argument("--threshold", type=float, default=200.0)
    p.add_argument("--window", type=int, default=100)
    a = p.parse_args()

    dirs = sorted({f.parent for f in Path(a.root).rglob("events.out.tfevents.*") if a.filter in str(f.parent)})
    print(f"| run | steps | conv@{a.threshold:g} | final mean{a.window} | trunc% | eval 100ep |")
    print("|---|---|---|---|---|---|")
    for d in dirs:
        r = summarize(d, a.threshold, a.window)
        if r is None:
            continue
        conv = f"{r['conv_step']:,}" if r["conv_step"] else "미도달"
        ev = f"{r['eval']:.1f}" if r["eval"] is not None else "-"
        print(f"| {r['run']} | {r['last_step']:,} | {conv} | {r['final_mean100']:.1f} | "
              f"{100 * r['trunc_frac']:.0f} | {ev} |")


if __name__ == "__main__":
    main()
