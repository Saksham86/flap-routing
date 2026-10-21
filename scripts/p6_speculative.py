"""P6 token-level cascade: speculative (assisted) decoding with a compressed DRAFT of the same model.

The draft (pruned and/or int8) proposes tokens; the full model verifies them in one forward pass and keeps
the agreed prefix. With greedy decoding the output must be IDENTICAL to the full model's -- checked per prompt.

Measurement design (robust to CPU frequency drift, which spoiled earlier CPU timings):
  * for every prompt, the baseline (full model alone) and the speculative run are timed back to back,
    alternating which goes first; the per-prompt RATIO is the measurement;
  * report: median speedup with a bootstrap 95% CI, tokens generated per full-model forward pass,
    identical-output fraction, and the power state (CPU) before/after.
Prompts: GSM8K test-split questions from the query pool ("Question: ...\\nAnswer:"), generation is where the
time goes. Generation length is fixed (EOS suppressed) so both runs produce the same number of tokens.

Outputs in results/p6/<run_id>/: speculative.csv (per prompt x draft), speculative.json (summary)
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from qacr.common.config import get_dotted, load_config  # noqa: E402
from qacr.common.logging_utils import close_logger, get_logger  # noqa: E402
from qacr.common.run import start_run  # noqa: E402
from qacr.common.seed import seed_everything  # noqa: E402
from qacr.pruning.flap import PrunePlan, apply_plan  # noqa: E402
from qacr.runtime.variants import parse_dtype  # noqa: E402


class ForwardCounter:
    def __init__(self, model):
        self.n = 0
        self.h = model.register_forward_hook(lambda *a: self._inc())

    def _inc(self):
        self.n += 1


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    m = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg["seed"])
    run = start_run(cfg, run_id=args.run_id, argv=sys.argv)
    log = get_logger("p6.spec", run.path("run.log"))
    device = cfg["device"]
    threads = get_dotted(cfg, "threads")
    if threads:
        torch.set_num_threads(int(threads))
    power = None
    if device == "cpu":
        sys.path.insert(0, str(REPO / "scripts"))
        from p1_profile import power_state
        power = power_state
        ps = power()
        if get_dotted(cfg, "require_ac_power", True) and ps.get("on_ac_power") is False:
            log.error("on battery: refusing to time on CPU")
            close_logger(log)
            return 3

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["base"])
    target = AutoModelForCausalLM.from_pretrained(cfg["base"], dtype=parse_dtype(cfg["target_dtype"])).to(device).eval()
    pool = pd.read_parquet(REPO / cfg.get("pool_path", "data/p2/query_pool.parquet"))
    qs = pool[(pool["task"] == "gsm8k") & (pool["split"] == "test")].sort_values("query_id").head(cfg["n_prompts"])
    prompts = [tok(f"Question: {q}\nAnswer:", return_tensors="pt").input_ids.to(device) for q in qs["question_text"]]
    n_new = int(cfg["max_new_tokens"])
    eos = target.generation_config.eos_token_id
    kw = dict(max_new_tokens=n_new, min_new_tokens=n_new, do_sample=False, pad_token_id=eos if isinstance(eos, int) else 0)

    rows, summary = [], {}
    for dspec in cfg["drafts"]:
        name = dspec["name"]
        draft = AutoModelForCausalLM.from_pretrained(cfg["base"], dtype=parse_dtype(dspec.get("dtype", cfg["target_dtype"])))
        draft = draft.to(device).eval()
        if dspec.get("plan"):
            apply_plan(draft, PrunePlan.load(REPO / dspec["plan"] if not Path(dspec["plan"]).is_absolute()
                                             else dspec["plan"]), physical=True)
        if dspec.get("int8"):
            from qacr.runtime.fast import quantize_int8_cpu
            draft = quantize_int8_cpu(draft)
        gc = draft.generation_config
        gc.num_assistant_tokens = int(cfg["num_assistant_tokens"])
        gc.num_assistant_tokens_schedule = cfg["schedule"]
        gc.assistant_confidence_threshold = float(cfg["assistant_confidence_threshold"])
        # warm-up both paths once
        with torch.no_grad():
            target.generate(prompts[0], **dict(kw, max_new_tokens=8, min_new_tokens=8))
            target.generate(prompts[0], assistant_model=draft, **dict(kw, max_new_tokens=8, min_new_tokens=8))
        ratios, tpf, same = [], [], []
        for i, ids in enumerate(prompts):
            res = {}
            order = ("base", "spec") if i % 2 == 0 else ("spec", "base")
            for mode in order:
                cnt = ForwardCounter(target)
                sync(device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    if mode == "base":
                        out = target.generate(ids, **kw)
                    else:
                        out = target.generate(ids, assistant_model=draft, **kw)
                sync(device)
                res[mode] = (time.perf_counter() - t0, out[0, ids.shape[1]:].tolist(), cnt.n)
                cnt.h.remove()
            tb, ts = res["base"][0], res["spec"][0]
            identical = res["base"][1] == res["spec"][1]
            first_diff = next((k for k, (a, b) in enumerate(zip(res["base"][1], res["spec"][1])) if a != b),
                              min(len(res["base"][1]), len(res["spec"][1])))
            row = {"draft": name, "prompt": i, "query_id": qs.iloc[i]["query_id"], "base_s": tb, "spec_s": ts,
                   "speedup": tb / ts, "tokens": len(res["spec"][1]), "target_forwards_spec": res["spec"][2],
                   "tokens_per_target_forward": len(res["spec"][1]) / max(1, res["spec"][2]),
                   "identical": identical, "first_diff": first_diff, "order": "-".join(order)}
            if power:
                p = power()
                row.update({"on_ac": p.get("on_ac_power"), "cpu_perf": p.get("cpu_percent_performance")})
            rows.append(row)
            ratios.append(row["speedup"])
            tpf.append(row["tokens_per_target_forward"])
            same.append(identical)
            log.info("[%s] prompt %d: base %.1fs spec %.1fs -> x%.2f | %.2f tok/target-fwd | identical=%s",
                     name, i, tb, ts, row["speedup"], row["tokens_per_target_forward"], identical)
        lo, hi = boot_ci(ratios)
        summary[name] = {"median_speedup": float(statistics.median(ratios)), "speedup_ci95": [lo, hi],
                         "mean_tokens_per_target_forward": float(np.mean(tpf)),
                         "identical_fraction": float(np.mean(same)), "n_prompts": len(ratios), "draft_spec": dspec}
        log.info("[%s] SUMMARY median speedup x%.2f (95%% CI %.2f-%.2f), %.2f tokens/target forward, identical %.0f%%",
                 name, summary[name]["median_speedup"], lo, hi, summary[name]["mean_tokens_per_target_forward"],
                 summary[name]["identical_fraction"] * 100)
        run.write_json("speculative.json", {"summary": summary, "config_note": "greedy; EOS suppressed"})
        del draft
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    with run.path("speculative.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    close_logger(log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
