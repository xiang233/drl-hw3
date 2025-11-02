# data/plotenvsteps.py
import argparse, glob, os, sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_xy(run_dir: str, metric: str):
    ea = EventAccumulator(run_dir)
    ea.Reload()
    # pull Y series for the desired metric and X as env steps
    y_series = ea.Scalars(metric)
    x_series = ea.Scalars("Train_EnvstepsSoFar")
    if not y_series or not x_series:
        raise KeyError(f"Missing tag(s) in {run_dir}. "
                       f"Found tags: {sorted(set(list(ea.Tags().get('scalars', []))))}")
    env_by_step = {p.step: p.value for p in x_series}
    xs, ys = [], []
    for s in y_series:
        if s.step in env_by_step:
            xs.append(env_by_step[s.step])
            ys.append(s.value)
    return np.array(xs), np.array(ys)

def smooth_curve(y, alpha):
    if alpha <= 0 or len(y) == 0:
        return y
    out = np.zeros_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * out[i-1] + (1 - alpha) * y[i]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Run dirs and/or glob patterns. Pass multiple items separated by spaces.")
    ap.add_argument("--metric", required=True, help="Scalar tag, e.g., Eval_AverageReturn or 'Baseline Loss'")
    ap.add_argument("--out", required=True, help="Output image path (PNG)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--smooth", type=float, default=0.0, help="EMA alpha in [0,1). 0 = no smoothing")
    ap.add_argument("--ylim", nargs=2, type=float, default=None)
    args = ap.parse_args()

    # Expand globs
    run_dirs = []
    for pat in args.runs:
        matches = sorted(glob.glob(pat))
        if matches:
            run_dirs.extend(matches)
    # Dedup while preserving order
    seen = set()
    run_dirs = [d for d in run_dirs if not (d in seen or seen.add(d))]

    if not run_dirs:
        print(f"No run dirs matched any of: {args.runs}", file=sys.stderr)
        sys.exit(1)

    # Ensure output dir exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    plt.figure(figsize=(9, 5))
    for rd in run_dirs:
        try:
            xs, ys = load_xy(rd, args.metric)
        except KeyError as e:
            print(str(e), file=sys.stderr)
            continue
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        ys_s = smooth_curve(ys, args.smooth)
        label = os.path.basename(rd)
        plt.plot(xs, ys_s, label=label)

    plt.xlabel("Environment Steps")
    plt.ylabel(args.metric)
    if args.title:
        plt.title(args.title)
    if args.ylim:
        plt.ylim(args.ylim[0], args.ylim[1])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
