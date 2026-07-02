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


def probe(X, y, seed=0):
    """5-fold CV ROC-AUC of an L2 logistic probe; plus label-shuffled chance."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    if len(set(y.tolist())) < 2 or min(np.bincount(y)) < 5:
        return (float("nan"), float("nan"), int(y.sum()), int(len(y) - y.sum()))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean()
    rng = np.random.default_rng(seed)
    ysh = y.copy(); rng.shuffle(ysh)
    auc_sh = cross_val_score(clf, X, ysh, cv=cv, scoring="roc_auc").mean()
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
