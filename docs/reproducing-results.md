# Reproducing the Results

This page lists the commands used to reproduce the results reported for **Déjà Cue: Localizing States in Object Histories via Vocabulary-Relative Coordinates**. Run every command from the repository root after completing the [installation steps](installation.md).

## Quick Reproduction

The complete CPU workflow verifies the download, reproduces every fixed-feature result, recalculates the statistical summaries, and runs the test suite:

```bash
python scripts/reproduce_all.py --device cpu
```

This workflow does not require raw video, masks, encoder weights, or learned checkpoints.

## VOST Results

| Result | Command | Output |
| --- | --- | --- |
| Select the 100-history sample and confirm the final 78-history cohort | `python scripts/inspect_vost_cohort.py` | Printed cohort summary |
| Compare the seven fixed-feature methods | `python scripts/reproduce_main.py --device cpu` | `results/main_reproduction.json` |
| Reproduce adaptive peak expansion, prompt variants, duration scoring, and candidate ranking | `python scripts/reproduce_extended.py --device cpu` | `results/extended_reproduction.json` |
| Recalculate confidence intervals and statistical tests | `python scripts/check_reference_results.py` | Printed statistical summary |
| Recalculate robustness experiments | `python scripts/check_robustness_results.py` | Printed summaries for each experiment |

The main comparison reproduces 2,184 predicted windows. The extended evaluation reproduces another 4,992 windows. Both commands compare every prediction and summary statistic with the included reference files.

The complete VOST results require both `reproduce_main.py` and `reproduce_extended.py`: the first covers the seven fixed-feature methods, while the second covers adaptive peak expansion and the additional analyses.

## Seven-History Results

Run:

```bash
python scripts/reproduce_seven_history.py --device cpu
```

The command evaluates 7 histories, 16 states, 32 descriptions, and 59 reference episodes under four coordinate choices. It reproduces all 128 predicted windows and writes `results/seven_history_reproduction.json`.

The learned-decoder summaries are recalculated with:

```bash
python scripts/check_reference_results.py
```

The included results cover Moment-DETR, QD-DETR, EaTR, CG-DETR, UVCOM, TR-DETR, TaskWeave, and Sim-DETR over seeds 3407, 3408, and 3409. To repeat training, follow the commands in [Reproducibility Protocol](reproducibility.md#6-native-learned-decoders).

## Additional Experiments

| Experiment | Command | Included results |
| --- | --- | --- |
| Synthetic duration scoring | `python scripts/check_robustness_results.py` | 270 design settings across three scoring methods |
| Missing observations and identity switches | Same command | Five perturbation settings over three seeds |
| Recurrent states | Same command | D-NeRF predictions and references |
| Development-set controls | Same command | Five held-out folds, state permutations, and temporal shifts |
| Annotation agreement and sensitivity | Same command | Three histories and three annotation sets |
| VOST paraphrase consistency | Same command | Absolute and vocabulary-relative results over 78 histories |

The reference JSON files are stored in `data/reference/`. They contain the predictions and summary rows needed for these calculations, without large intermediate score tensors.

## Metrics

Temporal IoU uses inclusive integer intervals.

- R@1 at threshold `tau` records whether the selected interval reaches tIoU `tau`.
- Top-1 tIoU records the overlap of the selected interval itself.
- VOST averages descriptions within states, states within histories, histories from the same source together, and then source videos uniformly.
- The seven-history evaluation averages descriptions within states, states within histories, and histories uniformly. Recurrent states use the best matching annotated occurrence.

## Scope and Limitations

The CPU workflow reproduces retrieval from the included features, interval selection, metrics, and statistical calculations. It does not repeat human annotation, raw-video preprocessing, encoder inference, learned optimization, or the RTX 4090 timing measurement.

The VOST comparison is designed to isolate the effect of the query coordinate while keeping features and temporal search fixed. It should not be read as a general leaderboard comparison. The seven-history collection is small, its three independently held-out histories show mixed effects, and none of the eight learned-decoder comparisons remains significant after Holm correction.

## Verify the Download

Run:

```bash
python scripts/verify_package.py
```

This command checks the file list and SHA-256 values, then validates the data shapes, feature values, history mappings, query order, reference intervals, and experiment files.
