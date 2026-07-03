#!/usr/bin/env python3
"""Workstream B add-on e3: hidden-state success linear probe.

For each baseline checkpoint, load the checkpoint, re-encode its own train-set
rollout transitions, extract the action-end and observation-end last-layer hidden
states, and fit an L2 logistic-regression probe (5-fold CV, ROC-AUC) that predicts
episode success from those hidden states. A label-shuffled control gives the chance
level. This asks: is trajectory success linearly decodable from the policy's hidden
state, and does that decodability grow with training?

Runs on GPU (does its own forward; independent of the CE/cosine scorer). Reuses the
scorer's model-loading + transition-encoding helpers read-only.
"""
import argparse, csv, glob, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wm_score_transition_dump as S  # read-only reuse

STEP_ORDER = ["init", "15", "30", "45", "60", "75", "90", "105", "120", "135", "150"]


def extract_hiddens(model, encoded, device, batch_size, max_n):
    import torch
    A, O, y = [], [], []
    use_cuda = device.startswith("cuda")
    for start in range(0, len(encoded), batch_size):
        if len(y) >= max_n:
            break
        batch = encoded[start:start + batch_size]
        input_ids = torch.tensor([it.input_ids for it in batch], dtype=torch.long, device=device)
        attn = torch.tensor([it.attention_mask for it in batch], dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda):
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False, output_hidden_states=True)
        last = out.hidden_states[-1]
        for i, it in enumerate(batch):
            if it.action_end_pos is None or it.obs_end_pos is None:
                continue
            er = it.row.get("episode_rewards")
            try:
                lab = 1 if float(er) > 0 else 0
            except Exception:
                continue
            A.append(last[i, it.action_end_pos].float().cpu().numpy())
            O.append(last[i, it.obs_end_pos].float().cpu().numpy())
            y.append(lab)
    return np.asarray(A, np.float32), np.asarray(O, np.float32), np.asarray(y, np.int64)


def _rankdata(a):
    """Ranks with ties averaged (scipy.stats.rankdata equivalent), pure numpy."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def _auc(scores, y):
    """ROC-AUC via the Mann-Whitney U rank statistic (handles ties)."""
    npos = int(y.sum()); nneg = int(len(y) - npos)
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def _fit_logreg_torch(X, y, C=1.0, epochs=400, lr=0.05):
    """L2-regularised logistic regression via Adam (sklearn-free). CPU torch."""
    import torch
    Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    n, d = Xt.shape
    w = torch.zeros(d, requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    l2 = 1.0 / (C * n)
    for _ in range(epochs):
        opt.zero_grad()
        z = Xt @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt) + l2 * (w * w).sum()
        loss.backward(); opt.step()
    return w.detach().numpy(), float(b.detach().numpy()[0])


def _stratified_folds(y, k, seed):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in (0, 1):
        idx = np.where(y == cls)[0]; rng.shuffle(idx)
        for i, ix in enumerate(idx):
            folds[i % k].append(int(ix))
    return [np.array(sorted(f), dtype=np.int64) for f in folds]


def _cv_auc(X, y, k=5, seed=0):
    folds = _stratified_folds(y, k, seed)
    aucs = []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        mu = X[tr].mean(0, keepdims=True); sd = X[tr].std(0, keepdims=True) + 1e-6
        Xtr = (X[tr] - mu) / sd; Xte = (X[te] - mu) / sd
        w, b = _fit_logreg_torch(Xtr, y[tr].astype(np.float32))
        a = _auc(Xte @ w + b, y[te])
        if not np.isnan(a):
            aucs.append(a)
    return float(np.mean(aucs)) if aucs else float("nan")


def probe(X, y, seed=0):
    """5-fold CV ROC-AUC of an L2 logistic probe; plus label-shuffled chance.

    sklearn-free: standardization + torch logistic regression + rank-based AUC.
    """
    if len(set(y.tolist())) < 2 or min(np.bincount(y)) < 5:
        return (float("nan"), float("nan"), int(y.sum()), int(len(y) - y.sum()))
    auc = _cv_auc(X, y, seed=seed)
    rng = np.random.default_rng(seed)
    ysh = y.copy(); rng.shuffle(ysh)
    auc_sh = _cv_auc(X, ysh, seed=seed)
    return (float(auc), float(auc_sh), int(y.sum()), int(len(y) - y.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--ckpt-root", required=True, help="baseline checkpoints/<exp> dir")
    ap.add_argument("--dump-root", required=True, help="bdiag_rollouts/<exp>")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-n", type=int, default=4000, help="cap transitions per checkpoint")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path)

    rows_out = []
    for step in STEP_ORDER:
        dd = os.path.join(args.dump_root, f"step{step}")
        dumps = sorted(glob.glob(os.path.join(dd, "*.wm_transitions.jsonl")))
        if not dumps:
            continue
        rows = [json.loads(l) for l in open(dumps[0])]
        spec_path = "base" if step == "init" else os.path.join(args.ckpt_root, f"global_step_{step}")
        spec = S.CheckpointSpec(label=f"{args.exp}_step{step}", path=spec_path)
        print(f"PROBE_LOAD step={step} ckpt={spec_path} n_rows={len(rows)}", flush=True)
        model = S.load_model(args.model_path, spec, device=args.device, dtype_name=args.dtype)
        encoded = S.encode_transitions(rows, tok, args.max_length)
        A, O, y = extract_hiddens(model, encoded, args.device, args.batch_size, args.max_n)
        del model
        import torch, gc
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        gc.collect()
        for feat_name, X in (("action_hidden", A), ("obs_hidden", O), ("action+obs", np.concatenate([A, O], axis=1) if len(A) else A)):
            auc, auc_sh, ns, nf = probe(X, y) if len(X) else (float("nan"),) * 2 + (0, 0)
            rows_out.append({"exp": args.exp, "step": step, "feature": feat_name,
                             "probe_auc": auc, "chance_auc": auc_sh, "n_succ": ns, "n_fail": nf})
            print(f"PROBE step={step} feat={feat_name} auc={auc:.4f} chance={auc_sh:.4f} n={ns}/{nf}", flush=True)

    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    print("BDIAG_PROBE_DONE out_csv=%s" % args.out_csv)


if __name__ == "__main__":
    main()
