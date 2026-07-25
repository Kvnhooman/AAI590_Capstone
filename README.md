# Modeling pipeline — Cross-Domain NER under Label Scarcity

This folder holds the full, ordered notebook pipeline for the capstone. It answers one
practical question: **given a fixed annotation budget in a new domain, which adaptation
strategy should you use?** We train a `bert-base-cased` NER baseline on **CoNLL-2003**
(newswire) and adapt it to two target domains — **WNUT-17** (social media) and **SciERC**
(scientific abstracts) — under matched budgets of **50 / 100 / 200** labeled sentences and
fixed seeds **13 / 42 / 101**.

Three adaptation strategies (the paper's three "arms") are compared:

| Arm | Strategy | Notebook |
|-----|----------|----------|
| 1  | Few-shot fine-tuning | `05` (+ `06a`→`06b`, a mitigation ablation) |
| 2  | Domain-adaptive pretraining (DAP) + few-shot fine-tuning | `07` → `08` |
| 3  | LLM prompting, no fine-tuning | `09` |

## Results at a glance

The full pipeline has been run end-to-end; all outputs live under `results/` on Drive.

- **Baseline (04):** CoNLL-2003 in-domain F1 **0.91**; zero-shot cross-domain boundary F1 drops
  to **0.61** (WNUT) / **0.07** (SciERC) — the degradation the study is about.
- **Headline (typed F1, budgets 50→200):** under tiny label budgets, **LLM prompting (Arm 3,
  `gpt-4o-mini`) beats both fine-tuning arms on both domains** and is roughly *flat* with budget,
  while fine-tuning *rises* — a clear data-efficiency crossover.

  | | WNUT 50/100/200 | SciERC 50/100/200 |
  |---|---|---|
  | Few-shot FT (Arm 1) | 0.30 / 0.35 / 0.40 | 0.15 / 0.24 / 0.37 |
  | DAP + FT (Arm 2) | 0.15 / 0.33 / 0.38 | 0.11 / 0.22 / 0.36 |
  | **LLM prompting (Arm 3)** | **0.45 / 0.45 / 0.46** | **0.40 / 0.43 / 0.44** |

- **DAP did not help:** Arm 2 ≈ Arm 1 (slightly lower), so domain-adaptive pretraining added no
  benefit at these budgets.
- **Per-type reversal (the key finding):** contrary to the hypothesis, the LLM is *strongest on
  the rare/specialized types* where fine-tuning collapses (SciERC `Metric` 0.41 vs 0.01, WNUT
  `product` 0.28 vs 0.04); fine-tuning only stays ahead on the single dominant type per domain
  (WNUT `person`, SciERC `Generic`).
- **Cost trade-off:** the LLM costs ~$0.16–1.06 per config (full-test *inference*) vs ~$0.01 for
  a fine-tuning run (one-time *training*) — different cost types, so fine-tuning amortizes far
  better at deployment scale. Total Arm 3 spend was ~$3.

> Numbers are typed micro-F1 on the full target test set; fine-tuning arms are means over seeds
> 13/42/101, the LLM arm is a single-seed (42) point estimate. See `summary_typed_f1_by_method_budget.csv`
> and `per_type_f1_*.csv` for the exact values.

## Run order

Run **top to bottom, 00 → 10**. The numbering is the dependency order: each notebook only
uses outputs written by lower-numbered notebooks. Everything reads/writes Google Drive at
`/content/drive/MyDrive/AAI590/data/processed` (set up by notebook 00).

```
00  →  01
        02  →  03  →  04  ┬─→ 05 ─┐
                          ├─→ 06a→06b │
                          └─→ 07 → 08 ─┼─→ 10
                              09 ───────┘
```

- `04` (baseline) is the hub: arms 1, 2, and DAP all start from it.
- `08` requires `07` (needs the DAP checkpoints).
- `10` requires every arm you want to appear in the final figures; missing arms are skipped
  with a warning, so you can run it early to check partial progress.
- `01` is analysis-only (nothing downstream depends on it).

## What each notebook does

| # | Notebook | Purpose | Reads | Writes |
|---|----------|---------|-------|--------|
| 00 | `00_collect_datasets_colab` | Download CoNLL-2003, WNUT-17, SciERC; convert to a common sentence-level JSONL schema (`tokens`, `tags`, `source_dataset`, `split`). | public URLs | `<ds>/<ds>_{train,validation,test}.jsonl`, `collection_report.json` |
| 01 | `01_eda_ner` | Exploratory analysis: entity-token rates, sentence/span lengths, vocab overlap vs CoNLL, rare-class and few-shot coverage checks. Produces the paper's Table 1 & 2 numbers. | processed JSONL | `eda_summary.json`, figures |
| 02 | `02_make_fewshot_splits` | Build the **deterministic** few-shot training subsets (50/100/200 × seeds 13/42/101) for WNUT-17 and SciERC. Reused unchanged by every fine-tuning/prompting arm. | `<ds>/<ds>_train.jsonl` | `fewshot_splits/<ds>/<ds>_train_{budget}_seed_{seed}.jsonl`, `manifest.csv` |
| 03 | `03_tokenize_and_align` | BERT subword tokenization + BIO label alignment (`-100` on continuation subwords / special tokens); builds per-dataset label maps. Tokenizes CoNLL (all splits), WNUT/SciERC (val+test), and every few-shot split. | processed + `fewshot_splits` | `tokenized/...`, `label_maps/<ds>_label_map.json` |
| 04 | `04_train_baseline` | Fine-tune `bert-base-cased` on CoNLL-2003 (lr 2e-5, 3 epochs). Report in-domain F1 and **zero-shot cross-domain** entity-boundary F1 on WNUT/SciERC. | `tokenized/conll2003`, label maps | `models/baseline_conll2003/`, `results/baseline_results.json` |
| 05 | `05_fewshot_finetune` | **Arm 1.** Transfer the baseline encoder + fresh head, few-shot fine-tune across the full 18-run grid (2 datasets × 3 budgets × 3 seeds). Reports typed F1, boundary F1, and per-type F1. | baseline, tokenized few-shot | `results/fewshot/` |
| 06a | `06a_mitigated_fewshot_finetune` | **Arm 1b — first attempt (superseded).** Catastrophic-forgetting mitigation with an over-conservative encoder LR (`5e-6`) + 2-epoch freeze. Underfit and scored *below* naive 05. Kept to document the thought process; not used in the final results. | baseline, tokenized few-shot | `results/mitigated/` |
| 06b | `06b_mitigated_fewshot_finetune_retuned` | **Arm 1b — retuned (the version used).** Same mitigation, but encoder LR raised to `2e-5` and freeze shortened to 1 epoch. Fixes the underfitting; result: it matches 05 but does *not* recover WNUT above zero-shot, so the regression is intrinsic, not a forgetting artifact. | baseline, tokenized few-shot | `results/mitigated_retuned/` |
| 07 | `07_domain_adaptive_pretraining` | **Arm 2, stage A.** Continue masked-LM pretraining of the baseline encoder on each target domain's **unlabeled** train text. Tracks perplexity + wall-clock. | baseline, raw target train text | `models/dap_<ds>/`, `results/dap_training_info.json` |
| 08 | `08_dap_fewshot_finetune` | **Arm 2, stage B.** Identical recipe to 05, but starting from the DAP encoder — isolating the single DAP variable. Prints an Arm 1 vs Arm 2 gain table. | `models/dap_<ds>`, tokenized few-shot | `results/dap_fewshot/` |
| 09 | `09_llm_prompting_eval` | **Arm 3.** Prompt an LLM (OpenAI `gpt-4o-mini`, with automatic prompt caching + rate-limit retry; free local fallback if no key) using the same few-shot examples as in-context demos, no training. Scored with the same metric on the full test set. Single demo seed (42). | raw test, `fewshot_splits` | `results/llm_prompting/` |
| 10 | `10_results_aggregation` | Combine every arm into the headline deliverables: **data-efficiency curve**, **cost-performance** plot, and **per-entity-type** breakdown. | all `results/*` | `combined_results.csv`, `summary_*.csv`, `*.png`, `per_type_f1_*.csv` |

`ner_common_utils.py` is a legacy helper from an earlier version of the pipeline; the current
notebooks (00–10) are self-contained and do **not** import it. It's kept for reference only.

### Note on the two few-shot samplers (01 ↔ 02)

The few-shot sampling logic appears in **two** places on purpose:

- `01` — `sample_fewshot()` — **diagnostic only**, used to print the EDA coverage table.
- `02` — `stratified_sample_train()` — **authoritative**, writes the real `fewshot_splits/`.

They are intentionally the *same* algorithm (identical 80/20 entity/non-entity stratification,
seed, and shuffle order) so the EDA coverage numbers match the splits you actually train on.
**If you edit one, edit the other** — otherwise the EDA table silently stops describing the
real splits. Both function definitions carry a `KEEP IN SYNC` comment. (`01` runs before `02`,
so it re-simulates rather than reading `02`'s output — this keeps EDA/coverage ahead of split
creation in the pipeline order.)

Other repeated helpers (`load_jsonl`, entity-type/span extraction, the metric code) are
duplicated deliberately so each notebook runs standalone in Colab — repetition, not conflict.
The metric code in 05/06b/08/09 is byte-identical on purpose, so cross-arm F1 is comparable.

### Data processing lives in 00 → 02 → 03, not in 01

To avoid a common mix-up: `01` **does not process data**. It only reads the standardized JSONL
from `00` and writes `eda_summary.json` (stats). The transforms the models depend on are:
`00` (raw → common JSONL) → `02` (few-shot splits) → `03` (tokenize + BIO-align).

## Consistency guarantees across the pipeline

- **One data schema:** every notebook consumes the common JSONL from 00.
- **One set of few-shot splits:** all arms train/demonstrate on the exact files from 02.
- **One metric:** typed micro-F1, boundary F1, and per-type F1 are computed with the same
  `seqeval` logic in 05, 06b, 08, and 09, so cross-arm numbers are directly comparable.
- **One evaluation set:** all arms evaluate on the **full** target test set (matched comparison).
- **One recipe for arms 1 & 2:** 08 mirrors 05's hyperparameters exactly; the only difference
  is the encoder starting point (DAP vs baseline).

## Before you run

1. **05 and 06b emit `per_type` and `train_seconds`** (needed by notebook 10 for the per-type
   table and cost analysis). Both have been run.
2. For **09**, supply an `OPENAI_API_KEY` (Colab secret, env var, temporary in-notebook paste,
   or the hidden `getpass` prompt). It uses OpenAI `gpt-4o-mini`; without a key it falls back to
   a weaker free local model (recorded per row, not interchangeable with the API results). The
   OpenAI account needs prepaid credit — a few dollars covers the full run.
3. **Mitigation result (06b):** the tuned mitigation matches naive few-shot (05) but does not
   lift WNUT above the zero-shot baseline — i.e. the WNUT regression is intrinsic to the domain,
   not a catastrophic-forgetting artifact. Report 06b as an ablation; keep 05 as the Arm 1 result.

## The 06a → 06b iteration (kept on purpose)

We deliberately keep **both** mitigation notebooks to show the debugging process rather than
just the final answer:

- **`06a`** — our first mitigation attempt. The encoder learning rate (`5e-6`) was too
  conservative and the encoder stayed frozen for 2 of 8 epochs, so the model **underfit** and
  scored *below* naive few-shot (05) — worst on SciERC.
- **`06b`** — after diagnosing the underfitting, we raised the encoder LR to `2e-5` and shortened
  the freeze to 1 epoch. This fixed the training, and the corrected result is what supports our
  conclusion: mitigation matches naive fine-tuning but cannot lift WNUT above the zero-shot
  baseline, so the WNUT regression is **intrinsic to the domain, not catastrophic forgetting**.

Only `06b` feeds the aggregation in notebook 10 (`results/mitigated_retuned/`); `06a` writes to
`results/mitigated/` and is illustrative only. The takeaway — a negative result reached through a
fix — is exactly what the ablation is meant to demonstrate.
