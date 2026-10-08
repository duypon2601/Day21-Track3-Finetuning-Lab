#!/usr/bin/env python3
"""Bonus B4 — a CONTROLLED rank sweep.

Placement is fixed at `text-linear`; learning rate, step budget, seed and loss mask are
the ones NB3's `correct` run used. Only `r` moves (alpha follows the 2r invariant), so
the difference between rows is rank and nothing else. r=16 is not retrained: it IS the
`correct` adapter, and its score is read from results/autopsy.json.

Writes results/rank_sweep.json. Scored on the target task, not on training loss.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labkit import data, evaluate as ev, generate, modeling, report, train  # noqa: E402
from labkit.config import SPECS, get_tier, training_epochs  # noqa: E402

RANKS = [8, 64]


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main() -> None:
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from trl import SFTConfig, SFTTrainer

    tier = get_tier()
    target = load_jsonl(ROOT / "data" / "eval_target.jsonl")
    train_rows = load_jsonl(ROOT / "data" / "split" / "train.jsonl")
    mask_mode = os.environ.get("MASK_MODE", "assistant-only")

    autopsy = json.loads((ROOT / "results" / "autopsy.json").read_text(encoding="utf-8"))
    correct = next(r for r in autopsy if r["run"] == "correct")
    runs = {r["run"]: r for r in report.read_rows("runs.csv", results_dir=ROOT / "results")}
    out = [{"r": 16, "run": "correct", "target": correct["target"],
            "format": correct["format"],
            "trainable_params": int(runs["correct"]["trainable_params"]),
            "final_loss": float(runs["correct"]["final_loss"]),
            "max_steps": int(runs["correct"]["max_steps"])}]

    for r in RANKS:
        key = f"rank_r{r}"
        spec = dataclasses.replace(SPECS["correct"], key=key, r=r, alpha=2 * r,
                                   label=f"all-linear · r={r} · LR 10x · 16-bit")
        adir = ROOT / "adapters" / key

        model, tok = generate.load_base(tier)
        rows = data.to_training_dataset(tok, train_rows, max_length=tier.max_length,
                                        mask_mode=mask_mode)
        targets = modeling.resolve_target_modules(model, spec.target)
        trainable = modeling.count_lora_params(model, targets, r)
        max_steps = train.planned_steps(len(rows), tier, training_epochs())
        sft_kwargs, _ = train.filter_kwargs(
            SFTConfig, train.sft_config_kwargs(tier, spec, str(adir), max_steps=max_steps))
        lora_kwargs, _ = train.filter_kwargs(
            LoraConfig, train.lora_config_kwargs(spec, targets))
        trainer = SFTTrainer(model=model, args=SFTConfig(**sft_kwargs),
                             train_dataset=Dataset.from_list(rows), processing_class=tok,
                             peft_config=LoraConfig(**lora_kwargs))
        train.align_trainable_precision(trainer.model)
        t0 = time.perf_counter()
        res = trainer.train()
        elapsed = time.perf_counter() - t0
        trainer.model.save_pretrained(adir)

        row = train.summarize_run(spec, tier, targets, trainable, elapsed,
                                  generate.peak_vram_gb())
        row["final_loss"] = round(res.training_loss, 4)
        row["max_steps"] = max_steps
        report.append_row(row, results_dir=ROOT / "results")
        del trainer, model
        generate.free_memory()

        # Score exactly as NB5 scores `correct`: fresh base + adapter, naive prompt.
        model, tok = generate.load_base(tier)
        model = PeftModel.from_pretrained(model, str(adir))
        model.eval()
        preds, _ = generate.generate_batch(model, tok, [x["input"] for x in target],
                                           system=generate.NAIVE_PROMPT, label=key)
        tgt = sum(ev.triage_field_accuracy(p, x["label"]) for p, x in zip(preds, target)) / len(target)
        fmt = sum(ev.has_required_keys(p, ev.TRIAGE_KEYS) for p in preds) / len(preds)
        del model
        generate.free_memory()

        out.append({"r": r, "run": key, "target": round(tgt, 4), "format": round(fmt, 4),
                    "trainable_params": trainable,
                    "final_loss": round(res.training_loss, 4), "max_steps": max_steps})
        print(f"{key}: target={tgt:.4f} format={fmt:.4f} loss={res.training_loss:.4f}")

    out.sort(key=lambda x: x["r"])
    report.write_json(out, "rank_sweep.json", results_dir=ROOT / "results")
    print(report.markdown_table(out))


if __name__ == "__main__":
    main()
