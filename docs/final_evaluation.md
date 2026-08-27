# Locked final evaluation protocol

M7 has an intentionally asymmetric workflow:

1. development-only results may be produced, compared, and repeated;
2. the final pipeline and its selection evidence are locked;
3. the frozen holdout is evaluated once;
4. the result and receipt become immutable study artifacts.

The implementation cannot prevent a person from deleting files or writing different software. It makes accidental reuse difficult, records the irreversible boundary, and fails closed when artifacts or configuration change.

## Current state

The lock, verification, and evaluation code is covered by unit tests and by an end-to-end synthetic CI run on Python 3.11 and 3.12. That synthetic run does not use TCGA data. The repository does not yet contain a real selected-candidate lock, a real holdout receipt, or TCGA final metrics.

Do not run the real \`evaluate\` command until the development comparison is finalized and its rationale has been reviewed.

## 1. Write the selection configuration

The selection configuration is a small JSON object. Example:

\`\`\`json
{
  "candidate_id": "elastic-net-1000",
  "primary_metric": "macro_f1",
  "selection_rationale": "Selected using development CV because it had the strongest macro F1 within uncertainty while retaining stable interpretable features.",
  "pipeline": {
    "family": "linear_gene_budget",
    "model": "elastic_net",
    "gene_budget": 1000,
    "negative_policy": "clip",
    "scaler": "standard",
    "seed": 20260825
  }
}
\`\`\`

The rationale must refer only to development analyses. Macro F1 is fixed as the primary metric. Supported locked families are:

- \`linear_gene_budget\`: L2 logistic, elastic net, or linear SVM with fold-tested gene budget and scaler;
- \`pca_logistic\`: PCA plus L2 logistic with a fixed component count;
- \`xgboost\`: the complete CPU/CUDA, thread, gene-budget, and tree-hyperparameter configuration.

Unknown fields are rejected. This is deliberate: a misspelled option must not silently fall back to a default.

## 2. Create the lock

\`\`\`bash
python -m tcga_ml.final_evaluation_cli lock \
  --config config/final_pipeline.json \
  --matrix data/cache/expression.float32.npy \
  --split data/processed/split_manifest.tsv \
  --genes data/cache/genes.tsv \
  --evidence results/benchmarks/classical/benchmark.json \
  --evidence results/feature_budget/elastic_net/feature_budget.json \
  --evidence results/benchmarks/xgboost-cpu/xgboost_benchmark.json \
  --output results/final/final_evaluation.lock.json
\`\`\`

At least one evidence file is required. Every evidence JSON must contain:

\`\`\`json
{
  "evaluation_scope": "development_cross_validation_only",
  "holdout_used": false
}
\`\`\`

The lock stores:

- the normalized candidate/pipeline configuration;
- the selection rationale and fixed primary metric;
- every evidence file's size and SHA-256;
- the selection configuration's size and SHA-256;
- the expression matrix, split manifest, and gene table sizes and SHA-256 values;
- a digest covering the complete lock payload;
- \`holdout_status: "sealed"\`.

The output is created with exclusive semantics and cannot overwrite an existing lock. Hashing the full expression cache can take time, but it binds the final run to the exact matrix rather than only its filename.

## 3. Verify without evaluating

\`\`\`bash
python -m tcga_ml.final_evaluation_cli verify \
  --lock results/final/final_evaluation.lock.json \
  --matrix data/cache/expression.float32.npy \
  --split data/processed/split_manifest.tsv \
  --genes data/cache/genes.tsv \
  --evidence results/benchmarks/classical/benchmark.json \
  --evidence results/feature_budget/elastic_net/feature_budget.json \
  --evidence results/benchmarks/xgboost-cpu/xgboost_benchmark.json
\`\`\`

Verification recomputes the lock digest and artifact hashes. Supplying \`--evidence\` rechecks the complete evidence SHA-256 set. It does not fit a model, predict samples, or compute holdout metrics.

## 4. Cross the one-time boundary

Only after the lock and rationale are final:

\`\`\`bash
python -m tcga_ml.final_evaluation_cli evaluate \
  --lock results/final/final_evaluation.lock.json \
  --matrix data/cache/expression.float32.npy \
  --split data/processed/split_manifest.tsv \
  --genes data/cache/genes.tsv \
  --evidence results/benchmarks/classical/benchmark.json \
  --evidence results/feature_budget/elastic_net/feature_budget.json \
  --evidence results/benchmarks/xgboost-cpu/xgboost_benchmark.json \
  --outdir results/final/locked-holdout \
  --receipt results/final/final_evaluation.receipt.json
\`\`\`

The receipt must be outside \`--outdir\`. Before holdout rows are loaded, the evaluator exclusively creates it with \`status: "holdout_access_started"\`. It also refuses any existing output directory. Therefore:

- a completed receipt blocks a second run;
- an in-progress or failed receipt also blocks an automatic retry;
- a failure after reservation is recorded as \`failed_after_holdout_reservation\`;
- recovery after such a failure requires a conscious methodological decision, not a new output path chosen silently.

After reservation, the evaluator checks that participants/cache indices are unique, development and holdout are both non-empty, and both contain the same cancer classes. It fits the locked pipeline on all development rows and predicts only the frozen holdout rows.

## Outputs

| File | Purpose |
| --- | --- |
| \`final_metrics.json\` | Locked candidate, sample counts, timing, macro/weighted F1, balanced accuracy, accuracy, per-class metrics, probability metrics when available, and confusion matrices |
| \`final_predictions.tsv\` | One row per holdout participant with truth, prediction, correctness, and class probabilities when supported |
| \`final_confusion.tsv\` | Raw counts and true-class-normalized fractions |
| \`final_pipeline.joblib\` | Fitted preprocessing, feature selection, and estimator pipeline |
| \`final_confusion.svg\` | Vector final-holdout confusion matrix |
| \`final_per_class_f1.svg\` | Vector per-class F1 chart |
| receipt JSON | Start/completion status and SHA-256/size records for every output |

For probabilistic models, the evaluator reports multiclass log loss and macro one-vs-rest ROC-AUC when defined. Linear SVM remains valid but reports probability metrics as unavailable rather than inventing calibrated probabilities.

The receipt and all outputs should be preserved together. The result is still retrospective TCGA classification and must not be described as prospective clinical validation.
