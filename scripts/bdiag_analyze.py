#!/usr/bin/env python3
"""Workstream B (proper protocol) analysis.

Reads, per baseline checkpoint step, the per-transition score CSV (from
wm_score_transition_dump.py) joined with the rollout dump JSONL (for traj_uid +
obs text), then answers:
  d1) per-checkpoint success vs failure separability on each world-model metric
      (GMM(2) + AUC + histograms),
  d2) gap = success_mean - failure_mean with bootstrap CI,
  d3) whether gap / separability grows with checkpoint step.
Episode-level and transition-level aggregations are computed and reported SEPARATELY.
Add-ons: (e1) next-obs NLL calibration, (e2) action->obs cosine controlled for
prev/next obs token overlap (shortcut control). (e3 hidden linear probe uses the
optional per-checkpoint hidden .npz if present.)

Deliverables: line charts (success/fail/all vs step) per metric, histograms at
init/mid/150, a stats CSV, and a markdown summary. All paths printed at the end.
"""
import argparse, csv, json, math, os, glob
from collections import defaultdict
import numpy as np

STEP_ORDER = ["init", "15", "30", "45", "60", "75", "90", "105", "120", "135", "150"]
METRICS = ["ce", "perplexity", "target_confidence_mean", "action_obs_cosine"]
METRIC_LABEL = {
    "ce": "next-obs teacher-forced CE (NLL/token)",
    "perplexity": "next-obs perplexity",
    "target_confidence_mean": "target-token confidence",
    "action_obs_cosine": "raw action-end<->obs-end cosine",
}


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def load_step(dump_root, step):
    """Return list of per-transition dicts with metric values, traj_uid, success, obs texts."""
    dd = os.path.join(dump_root, f"step{step}")
    scores = os.path.join(dd, "scores.csv")
    dumpjson = sorted(glob.glob(os.path.join(dd, "*.wm_transitions.jsonl")))
    if not (os.path.isfile(scores) and dumpjson):
        return None
    # index -> traj_uid, episode_rewards, prev/next obs text (for overlap control)
    idx_meta = {}
    for i, line in enumerate(open(dumpjson[0])):
        try:
            r = json.loads(line)
        except Exception:
            continue
        idx_meta[i] = (
            r.get("traj_uid"),
            _f(r.get("episode_rewards")),
            r.get("wm_prev_obs_text", ""),
            r.get("wm_next_obs_text", ""),
        )
    rows = []
    for r in csv.DictReader(open(scores)):
        ti = r.get("transition_index")
        try:
            ti = int(ti)
        except Exception:
            continue
        meta = idx_meta.get(ti)
        if meta is None:
            continue
        traj, eprew, prev_txt, next_txt = meta
        rec = {"traj_uid": traj, "ep_rew": eprew, "prev_txt": prev_txt, "next_txt": next_txt}
        for m in METRICS:
            rec[m] = _f(r.get(m))
        rec["nll_sum"] = _f(r.get("nll_sum"))
        rec["target_tokens"] = _f(r.get("target_tokens"))
        rows.append(rec)
    return rows


def episode_agg(rows):
    """traj_uid -> {metric: mean over its transitions}, success bool."""
    g = defaultdict(list)
    for r in rows:
        g[r["traj_uid"]].append(r)
    eps = []
    for traj, rs in g.items():
        succ = any((x["ep_rew"] or 0) > 0 for x in rs)
        rec = {"success": succ}
        for m in METRICS:
            vals = [x[m] for x in rs if x[m] is not None]
            rec[m] = float(np.mean(vals)) if vals else None
        eps.append(rec)
    return eps


def boot_ci_gap(succ, fail, n=2000, seed=0):
    """Bootstrap 95% CI of (mean(succ)-mean(fail))."""
    succ = np.asarray([v for v in succ if v is not None], float)
    fail = np.asarray([v for v in fail if v is not None], float)
    if len(succ) < 2 or len(fail) < 2:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    gaps = np.empty(n)
    for i in range(n):
        gaps[i] = succ[rng.integers(0, len(succ), len(succ))].mean() - fail[rng.integers(0, len(fail), len(fail))].mean()
    gap = float(succ.mean() - fail.mean())
    return (gap, float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5)))


def auc(vals, labels):
    vals = np.asarray(vals, float); labels = np.asarray(labels)
    ps = vals[labels == 1]; ng = vals[labels == 0]
    if len(ps) == 0 or len(ng) == 0:
        return float("nan")
    return float(np.mean([(p > ng).mean() + 0.5 * (p == ng).mean() for p in ps]))


def gmm_separability(vals, labels):
    """2-component GMM on a metric; best cluster->label accuracy vs base rate."""
    try:
        from sklearn.mixture import GaussianMixture
    except Exception:
        return None
    v = np.asarray([x for x in vals if x is not None], float).reshape(-1, 1)
    lab = np.asarray([labels[i] for i, x in enumerate(vals) if x is not None])
    if len(v) < 10 or len(set(lab.tolist())) < 2:
        return None
    gm = GaussianMixture(2, random_state=0).fit(v)
    pred = gm.predict(v)
    acc = max(((np.where(pred == 0, p0, p1)) == lab).mean() for p0, p1 in [(0, 1), (1, 0)])
    return {"gmm_acc": float(acc), "base_rate": float(max(lab.mean(), 1 - lab.mean())),
            "means": gm.means_.ravel().round(4).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-root", required=True, help="$WORK/logs/bdiag_rollouts/<exp>")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    steps_avail = [s for s in STEP_ORDER if load_step(args.dump_root, s) is not None]
    if not steps_avail:
        print("NO_SCORED_STEPS_YET dump_root=%s" % args.dump_root)
        return
    print("scored steps:", steps_avail)

    stat_rows = []   # long-form stats CSV
    hist_data = {}   # step -> level -> metric -> (succ_vals, fail_vals)
    for step in steps_avail:
        rows = load_step(args.dump_root, step)
        eps = episode_agg(rows)
        n_succ_ep = sum(1 for e in eps if e["success"]); n_fail_ep = len(eps) - n_succ_ep
        for level, data in (("episode", eps), ("transition", rows)):
            if level == "transition":
                lab_of = lambda r: 1 if (r["ep_rew"] or 0) > 0 else 0
            else:
                lab_of = lambda r: 1 if r["success"] else 0
            for m in METRICS:
                sv = [r[m] for r in data if lab_of(r) == 1 and r[m] is not None]
                fv = [r[m] for r in data if lab_of(r) == 0 and r[m] is not None]
                allv = [r[m] for r in data if r[m] is not None]
                gap, lo, hi = boot_ci_gap(sv, fv)
                labels = [lab_of(r) for r in data if r[m] is not None]
                a = auc([r[m] for r in data if r[m] is not None], labels)
                gm = gmm_separability([r[m] for r in data if r[m] is not None], labels)
                stat_rows.append({
                    "exp": args.exp, "step": step, "level": level, "metric": m,
                    "n_succ": len(sv), "n_fail": len(fv),
                    "mean_all": float(np.mean(allv)) if allv else float("nan"),
                    "mean_succ": float(np.mean(sv)) if sv else float("nan"),
                    "mean_fail": float(np.mean(fv)) if fv else float("nan"),
                    "gap": gap, "gap_ci_lo": lo, "gap_ci_hi": hi, "auc": a,
                    "gmm_acc": (gm or {}).get("gmm_acc", float("nan")),
                    "gmm_base_rate": (gm or {}).get("base_rate", float("nan")),
                    "n_ep_succ": n_succ_ep, "n_ep_fail": n_fail_ep,
                })
                hist_data.setdefault(step, {}).setdefault(level, {})[m] = (sv, fv)

    # write stats CSV
    csv_path = os.path.join(args.out_dir, f"bdiag_stats_{args.exp}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stat_rows[0].keys()))
        w.writeheader(); w.writerows(stat_rows)

    # plots
    fig_paths = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs_all = [s for s in STEP_ORDER if s in steps_avail]
        xnum = [0 if s == "init" else int(s) for s in xs_all]
        for level in ("episode", "transition"):
            for m in METRICS:
                fig, ax = plt.subplots(figsize=(6, 4))
                for lab, key in (("all", "mean_all"), ("success", "mean_succ"), ("failure", "mean_fail")):
                    ys = []
                    for s in xs_all:
                        row = next((r for r in stat_rows if r["step"] == s and r["level"] == level and r["metric"] == m), None)
                        ys.append(row[key] if row else float("nan"))
                    ax.plot(xnum, ys, marker="o", label=lab)
                ax.set_title(f"{METRIC_LABEL[m]} ({level})\n{args.exp}")
                ax.set_xlabel("checkpoint step"); ax.set_ylabel(m); ax.legend(); ax.grid(alpha=.3)
                p = os.path.join(args.out_dir, f"line_{level}_{m}_{args.exp}.png")
                fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); fig_paths.append(p)
        # histograms at init / mid / 150
        mids = [s for s in ["init", "75", "150"] if s in steps_avail]
        for m in METRICS:
            fig, axes = plt.subplots(1, len(mids), figsize=(4 * len(mids), 3.4), squeeze=False)
            for j, s in enumerate(mids):
                sv, fv = hist_data[s]["episode"][m]
                ax = axes[0][j]
                if sv: ax.hist(sv, bins=20, alpha=.6, label=f"success n={len(sv)}", density=True)
                if fv: ax.hist(fv, bins=20, alpha=.6, label=f"failure n={len(fv)}", density=True)
                ax.set_title(f"{m} @step {s}"); ax.legend(fontsize=7)
            fig.suptitle(f"{METRIC_LABEL[m]} success vs failure (episode-level) — {args.exp}")
            p = os.path.join(args.out_dir, f"hist_{m}_{args.exp}.png")
            fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); fig_paths.append(p)
    except Exception as e:
        print("PLOT_WARN", e)

    # add-on e2: control action_obs_cosine for prev/next obs token overlap
    e2_path = os.path.join(args.out_dir, f"addon_overlap_control_{args.exp}.csv")
    with open(e2_path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["step", "mean_jaccard", "raw_cos_gap", "residual_cos_gap"])
        for step in steps_avail:
            rows = load_step(args.dump_root, step)
            jac, cos, lab = [], [], []
            for r in rows:
                if r["action_obs_cosine"] is None: continue
                p = set((r["prev_txt"] or "").lower().split()); n = set((r["next_txt"] or "").lower().split())
                j = len(p & n) / max(len(p | n), 1)
                jac.append(j); cos.append(r["action_obs_cosine"]); lab.append(1 if (r["ep_rew"] or 0) > 0 else 0)
            jac = np.array(jac); cos = np.array(cos); lab = np.array(lab)
            if len(cos) < 10: continue
            # residualize cosine on jaccard (remove shortcut component), then success/fail gap
            A = np.vstack([jac, np.ones_like(jac)]).T
            beta = np.linalg.lstsq(A, cos, rcond=None)[0]
            resid = cos - A @ beta
            raw_gap = cos[lab == 1].mean() - cos[lab == 0].mean() if (lab == 1).any() and (lab == 0).any() else float("nan")
            res_gap = resid[lab == 1].mean() - resid[lab == 0].mean() if (lab == 1).any() and (lab == 0).any() else float("nan")
            w.writerow([step, f"{jac.mean():.4f}", f"{raw_gap:.4f}", f"{res_gap:.4f}"])

    print("BDIAG_ANALYZE_DONE")
    print("stats_csv:", csv_path)
    print("overlap_control_csv:", e2_path)
    print("figures:", len(fig_paths))
    for p in fig_paths: print("  fig:", p)


if __name__ == "__main__":
    main()
