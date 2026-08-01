# Modeling pipeline — Cross-Domain NER under Label Scarcity

This folder holds the full, ordered notebook pipeline for the capstone. It answers one
practical question: **given a fixed annotation budget in a new domain, which adaptation
strategy should you use?** We train a `bert-base-cased` NER baseline on **CoNLL-2003**
(newswire) and adapt it to target domains under matched budgets of **50 / 100 / 200** labeled
sentences and fixed seeds **13 / 42 / 101**. The core three-arm comparison runs on two target
domains — **WNUT-17** (social media) and **SciERC** (scientific abstracts) — and an extension
adds a third, **JNLPBA** (biomedical), as an external-validity check.

Three adaptation strategies (the paper's three "arms") are compared:

| Arm | Strategy | Notebook |
|-----|----------|----------|
| 1  | Few-shot fine-tuning | `05` (+ `06a`→`06b`, a mitigation ablation) |
| 2  | Domain-adaptive pretraining (DAP) + few-shot fine-tuning | `07` → `08` |
| 3  | LLM prompting, no fine-tuning | `09` |

## Results at a glance

The full pipeline has been run end-to-end; all outputs live under `results/` on Drive.

- **Baseline (04):** CoNLL-2003 in-domain F1 **0.91**; zero-shot cross-domain boundary F1 drops
  to **0.61** (WNUT) / **0.07** (SciERC) / **0.27** (JNLPBA) — the degradation the study is about.
- **Headline (typed F1, budgets 50→200):** under tiny label budgets, **LLM prompting
  (`gpt-4o-mini`) beats fine-tuning on all three domains**, and is roughly *flat* with budget
  while fine-tuning *rises* — a data-efficiency crossover replicated across domains.

  | | WNUT 50/100/200 | SciERC 50/100/200 | JNLPBA 50/100/200 |
  |---|---|---|---|
  | Few-shot FT (Arm 1) | 0.30 / 0.35 / 0.40 | 0.15 / 0.24 / 0.37 | 0.31 / 0.39 / 0.51 |
  | DAP + FT (Arm 2) | 0.15 / 0.33 / 0.38 | 0.11 / 0.22 / 0.36 | — |
  | **LLM prompting (Arm 3)** | **0.45 / 0.45 / 0.46** | **0.40 / 0.43 / 0.44** | **0.57 / 0.57 / 0.58** |

- **The crossover is consistent:** the LLM's lead is ~**+0.25** at budget 50 and narrows to
  ~**+0.06** at budget 200 in every domain — fastest on JNLPBA, the most learnable for fine-tuning.
- **DAP did not help:** Arm 2 ≈ Arm 1 (slightly lower) on WNUT/SciERC — even though DAP used the
  *full* unlabeled training split (thousands of sentences), extra in-domain text was not the
  limiting factor at these budgets.
- **Per-type reversal (a key finding):** contrary to the hypothesis, the LLM is *strongest on the
  rare/specialized types* where fine-tuning collapses (SciERC `Metric` 0.41 vs 0.01, WNUT
  `product` 0.28 vs 0.04, JNLPBA `RNA` 0.61 vs 0.03). Fine-tuning stays ahead only on the single
  dominant type per domain (WNUT `person`, SciERC `Generic`); on JNLPBA the LLM wins every type.
- **Contamination probe (12) — the LLM's edge is brittle:** when entities are replaced with novel
  character-perturbed surface forms (spans/types/context unchanged), the LLM's F1 *collapses* —
  WNUT 0.46→0.19, SciERC 0.44→**0.02** — and falls **below** the fine-tuning control on the same
  perturbed data (which drops far less). So the LLM's advantage rests substantially on
  recognizing *known* entity strings (plausibly pretraining exposure to these public benchmarks),
  and fine-tuning is the more robust choice for domains dominated by novel/emerging entities.
- **Cost trade-off:** the LLM costs ~$0.16–1.06 per config (ongoing per-inference API cost) vs
  ~$0.01 for a fine-tuning run (one-time training, then near-free inference) — so fine-tuning
  amortizes far better at deployment scale.

**Supplementary analyses (09b / 09c / 11):**

- **Zero-shot probe (09b):** on WNUT, zero-shot ≈ few-shot (demos add nothing → knowledge-driven);
  on SciERC, few-shot ≫ zero-shot (demos drive genuine in-context learning).
- **Retrieval demos (09c, GPT-NER-style):** kNN-selected demos beat fixed demos (WNUT 0.45→0.51)
  and are cheaper at large budgets.
- **Type-routed oracle hybrid (11):** routing each type to its best arm lifts typed F1 only
  ~0.03–0.04 over the LLM — it's already near the achievable ceiling.

> Numbers are typed micro-F1; WNUT/SciERC on the full test set (fine-tuning means over seeds
> 13/42/101; LLM single-seed 42). JNLPBA fine-tuning and LLM are both scored on a matched,
> representative **800-sentence** test subset (cost cap); JNLPBA has Arms 1 and 3 only. Exact
> values are in `results/**/…csv` and `results/three_domain/`.

## Run order

Run **top to bottom**. The numbering is the dependency order: each notebook only uses outputs
written by lower-numbered notebooks. Everything reads/writes Google Drive at
`/content/drive/MyDrive/AAI590/data/processed` (set up by notebook `00a`).

```
core:            00a → 01
                  02 → 03 → 04 ┬─→ 05 ─┐
                               ├─→ 06a → 06b │
                               └─→ 07 → 08 ─┼─→ 10
                                   09 ───────┘
extensions:      09 → 09b, 09c   ·   05/08/09 → 11   ·   05/09 → 12 (contamination)
biomedical (3rd domain):   00b → 13 → 14 → 15
```

- `04` (baseline) is the hub: arms 1, 2, and DAP all start from it.
- `08` requires `07` (DAP checkpoints); `10` aggregates whatever arms have run.
- `01` is analysis-only. All extension notebooks are optional and don't modify the core outputs.
- **Biomedical track:** `00b` collects JNLPBA; `13` does prep + zero-shot + fine-tuning; `14` runs
  the LLM arm; `15` unifies all three domains. These write only under `results/jnlpba/` and
  `results/three_domain/`, leaving the WNUT/SciERC results untouched.

## What each notebook does

| # | Notebook | Purpose | Writes |
|---|----------|---------|--------|
| 00a | `00a_collect_datasets_colab` | Download CoNLL-2003, WNUT-17, SciERC; convert to a common sentence-level JSONL schema. | `<ds>/<ds>_{train,validation,test}.jsonl` |
| 00b | `00b_collect_biomedical` | **Extension.** Download + standardize JNLPBA (biomedical; 5 types) into the same schema. | `jnlpba/jnlpba_*.jsonl` |
| 01 | `01_eda_ner` | EDA: entity-token rates, lengths, vocab overlap, few-shot coverage. Produces Table 1 & 2 numbers. | `eda_summary.json`, figures |
| 02 | `02_make_fewshot_splits` | Deterministic few-shot subsets (50/100/200 × seeds 13/42/101). | `fewshot_splits/…` |
| 03 | `03_tokenize_and_align` | BERT subword tokenization + BIO alignment; per-dataset label maps. | `tokenized/…`, `label_maps/…` |
| 04 | `04_train_baseline` | Fine-tune `bert-base-cased` on CoNLL-2003; in-domain + zero-shot cross-domain F1. | `models/baseline_conll2003/`, `results/baseline_results.json` |
| 05 | `05_fewshot_finetune` | **Arm 1.** Transfer encoder + fresh head, few-shot fine-tune (18-run grid). Typed/boundary/per-type F1. | `results/fewshot/` |
| 06a | `06a_mitigated_fewshot_finetune` | **Arm 1b — first attempt (superseded).** Mitigation with too-low encoder LR → underfit. Kept to document the process. | `results/mitigated/` |
| 06b | `06b_mitigated_fewshot_finetune_retuned` | **Arm 1b — retuned (used).** Matches 05 but doesn't recover WNUT above zero-shot → regression is intrinsic, not forgetting. | `results/mitigated_retuned/` |
| 07 | `07_domain_adaptive_pretraining` | **Arm 2a.** Continue MLM pretraining on unlabeled target text. | `models/dap_<ds>/`, `results/dap_training_info.json` |
| 08 | `08_dap_fewshot_finetune` | **Arm 2b.** Same recipe as 05 from the DAP encoder — isolates the DAP effect. | `results/dap_fewshot/` |
| 09 | `09_llm_prompting_eval` | **Arm 3.** LLM prompting (OpenAI `gpt-4o-mini`, prompt caching + retry; free local fallback). | `results/llm_prompting/` |
| 09b | `09b_llm_zeroshot` | **Extension — contamination/prior probe.** Arm 3 with no demonstrations. | `results/llm_prompting/` (budget 0) |
| 09c | `09c_llm_retrieval_demos` | **Extension — GPT-NER retrieval.** kNN-selected demos (MiniLM). | `results/llm_prompting/llm_retrieval_results.csv` |
| 10 | `10_results_aggregation` | WNUT/SciERC deliverables: data-efficiency curve, cost-performance, per-type. | `combined_results.csv`, `*.png`, … |
| 11 | `11_hybrid_analysis` | **Extension — oracle type-routed hybrid.** Head-room for a LinkNER-style combiner. | `hybrid_oracle_*.csv/png` |
| 12 | `12_contamination_counterfactual` | **Extension — contamination probe.** Novel entity perturbation; LLM vs fine-tuning drop on identical perturbed data. | `results/contamination/` |
| 13 | `13_jnlpba_prep_and_finetune` | **Biomedical.** JNLPBA splits + tokenize + zero-shot + fine-tuning (Arm 1), scored on a matched 800-subset (+ full-test reference). | `results/jnlpba/` |
| 14 | `14_jnlpba_llm` | **Biomedical.** JNLPBA LLM arm (Arm 3), 800-sentence cost cap. | `results/jnlpba/llm_jnlpba.csv` |
| 15 | `15_aggregate_all_domains` | **Biomedical.** Three-domain data-efficiency + gap + per-type tables (reads only). | `results/three_domain/` |

`ner_common_utils.py` is a legacy helper from an earlier version; the current notebooks are
self-contained and do **not** import it. It's kept for reference only.

### Note on the two few-shot samplers (01 ↔ 02)

The few-shot sampler appears in **two** places on purpose: `01`'s `sample_fewshot()` (diagnostic
only, for the EDA coverage table) and `02`'s `stratified_sample_train()` (authoritative, writes
the real `fewshot_splits/`). They use the *same* algorithm so the EDA numbers match the actual
splits. **If you edit one, edit the other** — both carry a `KEEP IN SYNC` comment. (`01` runs
before `02`, so it re-simulates rather than reading `02`'s output.)

Other repeated helpers (`load_jsonl`, span extraction, the metric code) are duplicated so each
notebook runs standalone in Colab — repetition, not conflict. The metric code across the
fine-tuning and LLM notebooks is byte-identical on purpose, so cross-arm F1 is comparable.

### Data processing lives in 00a → 00b → 02 → 03, not in 01

`01` does **not** process data — it reads the standardized JSONL and writes only stats. The
transforms the models depend on are: `00a`/`00b` (raw → common JSONL) → `02` (few-shot splits) →
`03` (tokenize + BIO-align).

## Consistency guarantees across the pipeline

- **One data schema:** every notebook consumes the common JSONL from `00a`/`00b`.
- **One set of few-shot splits:** all arms train/demonstrate on the exact files from `02`.
- **One metric:** typed micro-F1, boundary F1, and per-type F1 via the same `seqeval` logic
  everywhere, so cross-arm numbers are directly comparable.
- **Matched evaluation sets:** WNUT/SciERC arms evaluate on the full target test set; the JNLPBA
  fine-tuning and LLM arms both evaluate on the same fixed 800-sentence subset (seed 42).
- **One recipe for arms 1 & 2:** 08 mirrors 05 exactly; the only difference is the encoder start
  (DAP vs baseline).

## The 06a → 06b iteration (kept on purpose)

We deliberately keep **both** mitigation notebooks to show the debugging process, not just the
final answer. `06a` was our first attempt — the encoder learning rate (`5e-6`) was too
conservative and the encoder stayed frozen for 2 of 8 epochs, so it **underfit** and scored below
naive few-shot. `06b` raised the encoder LR to `2e-5` and shortened the freeze to 1 epoch, which
fixed training; the corrected result matches naive fine-tuning but still cannot lift WNUT above
the zero-shot baseline — evidence the WNUT regression is **intrinsic to the domain, not
catastrophic forgetting**. Only `06b` feeds the aggregation.

## Reproducing / re-running

1. **Keys:** `09`, `09b`, `09c`, `12`, `14` need an `OPENAI_API_KEY` (Colab secret, env var, or the
   hidden prompt — never hardcode/commit the key). They use `gpt-4o-mini`; a few dollars of
   prepaid credit covers all runs.
2. **GPU** is needed for the fine-tuning / DAP notebooks (04–08, 13); the aggregation notebooks
   (10, 11, 15) need neither GPU nor API.
3. Most training/LLM notebooks are **resumable** (skip completed dataset/budget/seed or cache
   per-sentence predictions), so an interrupted run picks up where it left off.
