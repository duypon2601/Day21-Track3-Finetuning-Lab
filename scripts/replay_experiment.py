#!/usr/bin/env python3
"""Follow-up to a FAILED regression gate: does 5% replay data fix the forgetting?

NB5's verdict for `correct` is not touched by this script and neither is any eval file.
This trains ONE extra adapter, `correct_replay`, identical to `correct` (placement,
rank, LR, seed, mask, and the SAME 58-step budget) except that data/replay_general.jsonl
-- 11 general Vietnamese Q&A pairs with no system prompt, none of them in
data/eval_regression.jsonl -- is mixed into the 225 training tickets (deck §6.3).

Scored on all four groups with the same code path as NB5. Writes
results/replay_experiment.json, including the raw regression answers.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labkit import data, evaluate as ev, generate, modeling, report, train  # noqa: E402
from labkit.config import SPECS, get_tier  # noqa: E402

KEY = "correct_replay"


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main() -> None:
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from trl import SFTConfig, SFTTrainer

    tier = get_tier()
    target = load_jsonl(ROOT / "data" / "eval_target.jsonl")
    regression = load_jsonl(ROOT / "data" / "eval_regression.jsonl")
    tickets = load_jsonl(ROOT / "data" / "split" / "train.jsonl")
    replay = load_jsonl(ROOT / "data" / "replay_general.jsonl")

    eval_questions = {r["instruction"] for r in regression}
    leaked = [r for r in replay if r["messages"][0]["content"] in eval_questions]
    assert not leaked, f"replay set leaks eval questions: {leaked}"

    mixed = tickets + replay
    random.Random(42).shuffle(mixed)

    runs = {r["run"]: r for r in report.read_rows("runs.csv", results_dir=ROOT / "results")}
    max_steps = int(runs["correct"]["max_steps"])       # same budget as `correct`
    spec = dataclasses.replace(SPECS["correct"], key=KEY,
                               label="all-linear · r=16 · LR 10x · 16-bit · +replay")
    adir = ROOT / "adapters" / KEY

    model, tok = generate.load_base(tier)
    rows = data.to_training_dataset(tok, mixed, max_length=tier.max_length)
    targets = modeling.resolve_target_modules(model, spec.target)
    trainable = modeling.count_lora_params(model, targets, spec.r)
    sft_kwargs, _ = train.filter_kwargs(
        SFTConfig, train.sft_config_kwargs(tier, spec, str(adir), max_steps=max_steps))
    lora_kwargs, _ = train.filter_kwargs(LoraConfig, train.lora_config_kwargs(spec, targets))
    trainer = SFTTrainer(model=model, args=SFTConfig(**sft_kwargs),
                         train_dataset=Dataset.from_list(rows), processing_class=tok,
                         peft_config=LoraConfig(**lora_kwargs))
    train.align_trainable_precision(trainer.model)
    t0 = time.perf_counter()
    res = trainer.train()
    elapsed = time.perf_counter() - t0
    trainer.model.save_pretrained(adir)

    row = train.summarize_run(spec, tier, targets, trainable, elapsed, generate.peak_vram_gb())
    row["final_loss"] = round(res.training_loss, 4)
    row["max_steps"] = max_steps
    report.append_row(row, results_dir=ROOT / "results")
    del trainer, model
    generate.free_memory()

    model, tok = generate.load_base(tier)
    model = PeftModel.from_pretrained(model, str(adir))
    model.eval()
    preds, lat = generate.generate_batch(model, tok, [r["input"] for r in target],
                                         system=generate.NAIVE_PROMPT, label=f"{KEY}/target")
    rpreds, _ = generate.generate_batch(model, tok, [r["instruction"] for r in regression],
                                        system=None, max_new_tokens=96,
                                        label=f"{KEY}/regression")
    tgt = sum(ev.triage_field_accuracy(p, r["label"]) for p, r in zip(preds, target)) / len(target)
    fmt = sum(ev.has_required_keys(p, ev.TRIAGE_KEYS) for p in preds) / len(preds)
    reg = sum(ev.keyword_recall(p, r["keywords"]) for p, r in zip(rpreds, regression)) / len(regression)
    scores = ev.GroupScores(target=tgt, regression=reg, format=fmt, latency_ms=lat, n=len(target))

    frozen = json.loads((ROOT / "results" / "baselines_frozen.json").read_text(encoding="utf-8"))
    base_b = ev.GroupScores(**{k: v for k, v in frozen["baseline_b"].items() if k != "extra"})
    verdict = ev.regression_gate(scores, base_b)

    report.write_json({
        "run": KEY,
        "n_tickets": len(tickets), "n_replay": len(replay),
        "replay_fraction": round(len(replay) / len(mixed), 4),
        "max_steps": max_steps, "final_loss": round(res.training_loss, 4),
        "scores": {"target": round(tgt, 4), "regression": round(reg, 4),
                   "format": round(fmt, 4), "latency_ms": round(lat, 1), "n": len(target)},
        "gate_vs_baseline_b": verdict.as_dict(),
        "regression_answers": [
            {"i": i, "question": r["instruction"], "pred": p,
             "score": ev.keyword_recall(p, r["keywords"])}
            for i, (r, p) in enumerate(zip(regression, rpreds))],
    }, "replay_experiment.json", results_dir=ROOT / "results")
    print(f"{KEY}: target={tgt:.4f} regression={reg:.4f} format={fmt:.4f} "
          f"-> {'PASSED' if verdict.passed else 'FAILED'}")
    for r in verdict.reasons:
        print(" -", r)


if __name__ == "__main__":
    main()
