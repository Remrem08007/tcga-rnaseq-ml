# Focused cancer-pair studies

M6 examines the two biologically close, pre-specified cancer pairs in the locked roadmap:

- LUAD vs LUSC;
- KIRC vs KIRP.

The purpose is to study where the development models struggle, which genes are repeatedly selected, and whether the fitted linear directions are stable across folds. This is retrospective classification of known TCGA labels. It is not a diagnostic or biomarker-validation study.

## Leakage boundary

The pair studies use the frozen split manifest through the same development-data loader as the earlier benchmarks. Only rows marked `development` are returned to the focused-pair runner. Pair filtering happens after that development-only load, so holdout participants are not passed to preprocessing, feature selection, fitting, prediction, error ranking, or model comparison.

Each pair uses stratified development cross-validation. For every fold:

1. the estimator is cloned;
2. `log2(x + 1)`, median imputation, zero-variance filtering, ANOVA gene selection, scaling, and the linear estimator are fit on the training fold;
3. the fitted fold pipeline predicts only its validation fold;
4. selected genes and class-specific coefficients are mapped back to the source TCGA gene table.

The runner assembles the fold predictions into one out-of-fold vector and asserts that every pair participant is covered exactly once. That protects the development error analysis from in-sample predictions. It does not replace the independent final test: M7 alone may evaluate the frozen holdout.

## Running the locked studies

```bash
python -m tcga_ml.focused_pairs_cli \
  --matrix data/cache/expression.float32.npy \
  --split data/processed/split_manifest.tsv \
  --genes data/cache/genes.tsv \
  --outdir results/focused_pairs/elastic_net \
  --model elastic_net \
  --gene-budget 1000 \
  --cv-folds 5 \
  --negative-policy clip \
  --n-jobs 0
```

Both pairs run by default. Repeat `--pair` to select explicit pairs and repeat `--model` to compare any of the locked linear estimators: `logistic_l2`, `elastic_net`, or `linear_svm`. Repeated options are rejected when they contain duplicates. The default gene budget is 1,000.

## Output contract

| File | Contents |
| --- | --- |
| `focused_pairs.json` | Complete configuration, metrics, fold summaries, OOF predictions, errors, gene rows, and hardest-pair ranking |
| `focused_pair_metrics.tsv` | Pair/model macro F1, balanced accuracy, accuracy, error count, and wall time |
| `focused_pair_confusion.tsv` | Raw confusion counts and fractions normalized within each true class |
| `focused_pair_predictions.tsv` | Exactly one OOF prediction per development participant in each pair/model study |
| `focused_pair_errors.tsv` | Misclassified OOF participants, sorted from largest to smallest absolute decision margin |
| `focused_pair_genes.tsv` | Fold selection frequency, class-specific mean coefficient, direction, and sign consistency mapped to source genes |

The JSON payload declares `evaluation_scope: development_cross_validation_only` and `holdout_used: false`. Those fields document the intended scope; the code-level protection is the development-only loader plus the exact-once OOF assertion.

## Reading the error analysis

The signed decision margin points toward the pair's second class: LUSC for LUAD/LUSC and KIRP for KIRC/KIRP. A positive value points toward that class and a negative value toward the first class. Absolute margin is useful for ranking confident errors within the same fitted model. It is not a calibrated probability, and its scale should not be compared casually across estimator families.

Use per-class recall and the row-normalized confusion table to distinguish asymmetric errors. For example, a high LUAD-to-LUSC fraction with a low LUSC-to-LUAD fraction indicates a directional confusion pattern that aggregate accuracy can hide.

## Reading the gene table

Selection frequency is the fraction of CV folds in which a gene survives fold-local selection. Sign consistency is the fraction of non-zero fold coefficients that agree in direction. The reported direction follows the mean coefficient toward one member of the pair.

These quantities are model-dependent predictive associations. They can be affected by correlated genes, cohort composition, preprocessing, regularization, and sampling variation. They do not establish causal biology, clinical utility, or a validated differential-expression signature. Biological interpretation should prioritize genes that combine high selection frequency, stable direction, and a plausible external literature basis, with external validation kept separate from model selection.
