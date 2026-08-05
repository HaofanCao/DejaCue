# Evaluation Guide

## Reproduce Reported Results

| Result | Included files | Command | What it checks |
| --- | --- | --- | --- |
| Downloaded files and data | `MANIFEST.json`, arrays, settings, and references | `python scripts/verify_package.py` | File names, SHA-256 values, data shapes, counts, paths, and feature values |
| VOST cohort selection | `data/reference/vost_cohort.json` | `python scripts/inspect_vost_cohort.py` | The 100 sampled histories, five-frame rule, 22 exclusions, earliest events, and H001-H078 mapping |
| Seven-method VOST comparison | `data/benchmark.json`, `data/features/siglip2/` | `python scripts/reproduce_main.py --device cpu` | All 2,184 predicted windows and summary metrics |
| Adaptive peak, prompts, normalization, ranking, and duration | VOST features and prompt variants | `python scripts/reproduce_extended.py --device cpu` | All 4,992 predicted windows and summary metrics |
| Seven-history coordinate study | `data/seven_history/` | `python scripts/reproduce_seven_history.py --device cpu` | All 128 predicted windows and summary metrics |
| Learned-decoder results | `seven_history_learned.json` and decoder code | `python scripts/check_reference_results.py` | 24 seed results, eight model summaries, intervals, sign tests, and Holm corrections |
| Seven-history controls | `seven_history_summary.json`, `vocabulary_stress.json` | Same command | Window schedules, duration controls, and description-duplication tests |
| Synthetic duration | `robustness/synthetic_duration.json` | `python scripts/check_robustness_results.py` | Results across 270 settings |
| Tracking perturbations | `robustness/tracking_perturbations.json` | Same command | Missing observations, distractor mixing, and identity switches |
| Recurrence | `robustness/dnerf_recurrence.json` | Same command | Temporal AP, recall, and precision |
| Development controls | `robustness/development_controls.json` | Same command | Five held-out folds, state permutations, and eight temporal shifts |
| Hard-negative margins | `robustness/hard_negative_margins.json` | Same command | Identity accuracy, history means, and bootstrap intervals |
| Annotation agreement and sensitivity | `robustness/annotation_agreement.json`, `annotation_sensitivity.json` | Same command | Agreement, consensus coverage, and metric changes across annotation sets |
| VOST paraphrase consistency | `robustness/vost_paraphrase.json` | Same command | Absolute and vocabulary-relative means over 78 histories |

Run every row and the test suite with:

```bash
python scripts/reproduce_all.py --device cpu
```

The complete VOST comparison requires both `reproduce_main.py` and `reproduce_extended.py`. The first covers seven fixed-feature methods; the second covers adaptive peak expansion and the additional analyses. See [Reproducing the Results](reproducing-results.md) for a shorter walkthrough.

## Recreate Data and Features

The following commands are needed only when recreating annotations, features, training runs, synthetic data, or timing measurements from additional inputs.

| Task | Command | Additional input | Output |
| --- | --- | --- | --- |
| Select the VOST cohort | `scripts/select_vost_cohort.py` | Included train and validation sequence lists | The seed-3407 sample and comparison with the final cohort |
| Compare two annotation passes | `scripts/create_vost_consensus_template.py` | Two complete, independent frame-label files | One row for each frame on which the annotators differ |
| Finalize VOST consensus | `scripts/finalize_vost_consensus.py` | Both annotation passes and resolved disagreements | A JSON summary containing input hashes, agreement, exclusions, and final intervals |
| Extract SigLIP 2 features | `scripts/extract_siglip2_features.py` | RGB frames, lineage masks, a source manifest, and local model files | 768-dimensional visual and text features plus a file list |
| Train a decoder | `scripts/train_learned_decoder.py` | Included five-history development features | An epoch-200 checkpoint and training summary |
| Evaluate a decoder | `scripts/evaluate_learned_decoder.py` | A generated checkpoint and included seven-history features | Description-, state-, history-, and collection-level results |
| Generate the synthetic duration study | `scripts/generate_synthetic_duration.py` | Included experiment settings | Synthetic predictions and summary metrics |
| Apply tracking perturbations | `deja_cue/experiments/perturbations.py` | Seven-history target and auxiliary tracks | Modified histories and a summary of the changes |
| Measure scan time | `scripts/benchmark_scan.py` | Included VOST or seven-history features | Warmed, synchronized milliseconds per query |

These commands check their inputs before writing new files. Feature extraction records the source and model hashes, annotation finalization records the input file hashes, and training saves the model settings with the checkpoint.

## Reference Files

- `data/reference/main.json` contains every primary predicted window, aggregate metric, bootstrap interval, and paired comparison.
- `data/reference/extended.json` contains the prompt, normalization, adaptive peak, ranking, and duration results.
- `data/reference/seven_history_summary.json` contains seven-history coordinate results and diagnostics.
- `data/reference/seven_history_learned.json` contains 24 decoder runs and the history-level values used for paired comparisons.
- `data/reference/robustness/` contains the rows needed to recalculate the robustness and sensitivity summaries.

A comparison stops with an error if a query is missing or duplicated, a predicted interval changes, a metric is not finite, or a value differs from its stored reference beyond the stated numerical tolerance.

## Output Files

The three numerical commands write:

```text
results/main_reproduction.json
results/extended_reproduction.json
results/seven_history_reproduction.json
```

Training checkpoints, decoder results, extracted features, synthetic results, annotation summaries, and timing measurements use paths chosen on the command line. Commands do not overwrite existing scientific results.

## Scope and Limitations

The included features are sufficient to reproduce retrieval, interval selection, metrics, and statistical calculations. Repeating encoder inference, human annotation, decoder training, synthetic generation, or timing requires the additional inputs listed above.

Third-party source and license files are retained under [`third_party/`](../third_party/).
