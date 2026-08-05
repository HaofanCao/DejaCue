# Data Guide

## Intended use

The repository data supports reproduction of the fixed-feature evaluation, file verification, statistical calculations, learned-decoder training on the five development histories, and software tests. It also provides the fixed protocol files and validation scripts needed to recreate data and features.

Histories use stable evaluation IDs H001-H078, and file references are relative to the package root. Result and benchmark records use documented formats for protocol inputs, metrics, and timing summaries.

## Bundled data groups

| Group | Contents | Supported use |
| --- | --- | --- |
| VOST source roster | Fixed train/validation sequence lists and the complete seed-3407 roster | Rebuild and verify the 100-history state-blind sample |
| VOST cohort | 100 sampled histories, 78 retained histories, 22 exclusions, final events, and H001-H078 mapping | Verify eligibility and reference-event selection |
| VOST evaluation | 78 histories, 156 states, 312 descriptions, reference intervals, and frozen features | Reproduce primary and extended evaluation |
| Prompt variants | Raw, photo, definite, and normalized three-form text features | Reproduce prompt sensitivity under fixed visual features |
| Development set | 5 histories, 13 states, 26 descriptions, 46 positive episodes, 86 positive training records, and 3,559 observed frames | Train the eight adapted decoders and reproduce protocol tests |
| Seven-history set | 7 histories, 16 states, 32 descriptions, 59 reference episodes, frozen target features, and 8 auxiliary tracks on four histories | Reproduce coordinate studies, learned evaluation, and identity diagnostics |
| Learned results | 24 model-seed aggregate rows and paired history-level rows | Recalculate means, sample standard deviations, intervals, and tests |
| Diagnostic records | Synthetic cells, tracking replicates, recurrence predictions, folds, shifts, margins, boundary-set comparisons, and paraphrase rows | Recalculate the additional experiments |

## VOST sampling and labels

`data/protocol/vost/train.txt` and `validation.txt` are the exact split lists consumed by `scripts/select_vost_cohort.py`. The sampler reads sequence names only. With seed 3407 it balances actions and target nouns in round-robin order and accepts at most one sequence per action-object pair.

The scientific protocol assigns two state descriptions before frame labeling. Two annotators then label every tracked-lineage frame independently as pre-state, transition, post-state, or unobserved while remaining blind to method scores. Every disagreement is resolved to a framewise consensus. A qualifying event has at least five stable frames on each side of the transition; the earliest event defines the designated pre- and post-state reference intervals.

`data/reference/vost_cohort.json` contains the final cohort used for evaluation. Together with the split lists, annotation formats, consensus validation, and feature manifests, it defines the 100-to-78 selection and the earliest-event intervals used for evaluation.

`deja_cue/vost_annotation.py` and the three VOST protocol scripts require complete passes, distinct annotator tokens, matching history rosters and identically ordered frame sets, exact disagreement resolution, immutable agreement frames, event derivation, and comparison to the bundled cohort.

## Frozen feature arrays

`data/features/siglip2/` contains one visual row per visible target frame and one text row per description. `data/features/siglip2_prompts/` contains the aligned prompt variants. All rows are finite, unit normalized, and 768-dimensional. Original temporal indices and visibility counts remain explicit; NumPy archives are opened with pickle disabled.

The seven-history and learned-development files use the same feature format. Loaders check paths, fields, row alignment, temporal order, dimensions, finite values, unit norms, state order, query order, and reference alignment before evaluation or training.

The encoder settings are stored in `configs/siglip2_encoder.json`. For every visible frame, the union of target-lineage masks defines a crop with 20% padding; pixels outside the union are set to 127 in each RGB channel. Text features are the normalized average of unit encodings for the raw description, `a photo of [description]`, and `the [description]`.

## Learned-decoder inputs and outputs

`data/learned/development/` contains the complete frozen inputs used to construct 86 positive run records. The seven-history evaluation store is never read by the training loader. The shared record builder supplies 768-dimensional visual features plus two temporal coordinates and vocabulary-relative text with full sibling context.

The package contains native training, objective, proposal-decoding, seed, checkpoint, and evaluation code for all eight decoder families. Decoder checkpoints are not included. `data/reference/seven_history_learned.json` contains the 24 run results used to recalculate the reported aggregates and paired statistics without repeating optimization.

Third-party source and license files are retained with the bundled code under [`third_party/`](../third_party/).

## Reference Results

`data/reference/main.json`, `extended.json`, and `seven_history_summary.json` contain exact selected windows and aggregate references. `data/reference/robustness/` retains only the structured rows needed to recompute synthetic duration, tracking perturbation, recurrence, development-control, hard-negative, boundary, and paraphrase summaries.

These files avoid large intermediate score tensors while retaining the rows needed to recalculate the reported results. Recreating features or training runs additionally requires the RGB, masks, encoder files, and checkpoints listed in the corresponding manifests.

## Inputs Not Included

- Native RGB frames and target-lineage mask arrays.
- Two complete independent VOST frame-label passes and the resolved framewise consensus when rerunning consensus finalization.
- The pinned SigLIP 2 model files listed in `configs/siglip2_encoder.json`.
- Learned-decoder checkpoints produced by the 24 training runs.
- Large rendering or intermediate-score files that are not needed to recalculate the included results.

The repository does not include the complete annotation passes or resolved consensus. It includes the validation scripts, final cohort information, and the agreement and annotation-sensitivity summaries.

Commands stop with an error when an annotation decision, file, checksum, reference alignment, or array field is missing or inconsistent.

The data tree combines metadata created for Déjà Cue with features derived from external research data. See [`DATA_LICENSE.md`](../DATA_LICENSE.md) before redistribution.

## Limitations

Fixed features isolate the reported retrieval and evaluation calculations from encoder execution. The extractor records hashes for raw inputs, model files, and software versions, but fresh extraction can still show hardware- or runtime-dependent numerical variation. The included learned results support recalculation of the reported summaries but do not repeat optimization. Timing depends on the executing hardware and is not expected to match the reported 15.2 ms exactly.

The method assumes target-lineage masks, a closed sibling vocabulary, and one selected interval per description. It has no calibrated null decision for an absent state. The main VOST cohort uses binary event-adjacent states and scores one designated occurrence even when a state recurs; the dataset contains 13 additional pre-state and 20 additional post-state occurrences outside those designated references. The seven-history set adds multi-state, interior, recurrent, and progressive cases, but it remains small and its three independently held-out histories show mixed coordinate effects.

Tracking stresses use synthetic feature mixing and switches; they do not model pixel-level mask drift or the full range of natural tracking failures. Results with other encoders, automatically discovered identities, open vocabularies, or broader multi-state collections have not been evaluated.
