#!/usr/bin/env python3
"""Save the raw generations behind the report's qualitative section.

NB2 and NB5 score baseline (b) and the fine-tune but keep only the fine-tune's target
predictions. The report needs both sides, on both eval groups, to show where the
fine-tune WINS (target) and where it LOSES (regression). Decoding is greedy, so this
reproduces the exact strings that were scored; the script re-scores them and asserts
the means match results/baselines_frozen.json and results/verdict.json.

Run AFTER NB5. Reads the frozen eval sets, changes nothing, writes
results/qualitative_detail.json.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labkit import evaluate as ev, generate, report  # noqa: E402
from labkit.config import get_tier  # noqa: E402


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main() -> None:
    from peft import PeftModel

    tier = get_tier()
    target = load_jsonl(ROOT / "data" / "eval_target.jsonl")
    regression = load_jsonl(ROOT / "data" / "eval_regression.jsonl")
    frozen = json.loads((ROOT / "results" / "baselines_frozen.json").read_text(encoding="utf-8"))
    verdict = json.loads((ROOT / "results" / "verdict.json").read_text(encoding="utf-8"))
    ft_row = verdict["comparison"][-1]

    model, tok = generate.load_base(tier)
    b_tgt, _ = generate.generate_batch(model, tok, [r["input"] for r in target],
                                       system=generate.OPTIMIZED_PROMPT, label="(b)/target")
    b_reg, _ = generate.generate_batch(model, tok, [r["instruction"] for r in regression],
                                       system=None, max_new_tokens=96, label="(b)/regression")
    model = PeftModel.from_pretrained(model, str(ROOT / "adapters" / "correct"))
    model.eval()
    c_tgt, _ = generate.generate_batch(model, tok, [r["input"] for r in target],
                                       system=generate.NAIVE_PROMPT, label="(c)/target")
    c_reg, _ = generate.generate_batch(model, tok, [r["instruction"] for r in regression],
                                       system=None, max_new_tokens=96, label="(c)/regression")

    tgt_rows = [{"i": i, "ticket": r["input"], "label": r["label"],
                 "b_pred": b, "b_score": ev.triage_field_accuracy(b, r["label"]),
                 "ft_pred": c, "ft_score": ev.triage_field_accuracy(c, r["label"])}
                for i, (r, b, c) in enumerate(zip(target, b_tgt, c_tgt))]
    reg_rows = [{"i": i, "question": r["instruction"], "keywords": r["keywords"],
                 "b_pred": b, "b_score": ev.keyword_recall(b, r["keywords"]),
                 "ft_pred": c, "ft_score": ev.keyword_recall(c, r["keywords"])}
                for i, (r, b, c) in enumerate(zip(regression, b_reg, c_reg))]

    def mean(rows, k):
        return sum(x[k] for x in rows) / len(rows)

    check = {
        "b_target": (mean(tgt_rows, "b_score"), frozen["baseline_b"]["target"]),
        "b_regression": (mean(reg_rows, "b_score"), frozen["baseline_b"]["regression"]),
        "ft_target": (mean(tgt_rows, "ft_score"), ft_row["target"]),
        "ft_regression": (mean(reg_rows, "ft_score"), ft_row["regression"]),
    }
    for name, (got, want) in check.items():
        print(f"{name}: regenerated {got:.4f} vs recorded {want:.4f}")
    reproduced = all(abs(g - w) < 1e-3 for g, w in check.values())

    report.write_json(
        {"reproduces_recorded_scores": reproduced,
         "check": {k: {"regenerated": round(g, 4), "recorded": round(w, 4)}
                   for k, (g, w) in check.items()},
         "target": tgt_rows, "regression": reg_rows},
        "qualitative_detail.json", results_dir=ROOT / "results")
    print("reproduces recorded scores:", reproduced)


if __name__ == "__main__":
    main()
