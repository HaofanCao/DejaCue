# Reproducibility Protocol

## 1. VOST Cohort Selection

The VOST sampler reads only sequence identifiers from the bundled training and validation split lists. With seed 3407, it creates a deterministic order within action and target-noun groups, draws histories in round-robin order, and admits at most one source sequence per action-object pair. It selects exactly 100 histories without reading labels, features, reference intervals, or method scores.

```text
python scripts/select_vost_cohort.py --output results/vost_selection.json
```

The output is checked against `data/reference/vost_cohort.json`. The final cohort has 78 qualifying histories and 22 histories with no qualifying event. Each retained history maps to one opaque evaluation ID, H001-H078.

## 2. Independent frame labels and consensus

Two state descriptions are fixed for each pre-state and post-state before frame labeling. Two annotators independently label every tracked-lineage frame as pre-state, transition, post-state, or unobserved while remaining blind to method scores. Every disagreement is resolved to consensus.

`deja_cue/vost_annotation.py` enforces this protocol. It requires complete and matching history rosters, identically ordered frame sets, distinct annotator tokens, valid labels, score blindness, and immutable agreement frames. A generated consensus template contains exactly the disagreement set. Final validation requires a decision for every disagreement and rejects edits to any frame where the two passes agreed.

```text
python scripts/create_vost_consensus_template.py --pass-a inputs/pass_a.json --pass-b inputs/pass_b.json --output results/consensus_template.json
python scripts/finalize_vost_consensus.py --pass-a inputs/pass_a.json --pass-b inputs/pass_b.json --consensus inputs/resolved_consensus.json --output results/consensus_summary.json
```

A qualifying event contains at least five stable pre-state frames followed by a transition and at least five stable post-state frames. If several events qualify, the earliest supplies the designated reference episodes. Finalization derives all events from consensus and requires an exact match to the bundled 100-to-78 partition and reference intervals.

The final cohort, evaluation labels, file formats, interval-building code, split lists, and validation scripts are included. Reproducing the fixed-feature results uses the final evaluation data. Repeating consensus validation uses the two annotation passes and resolved disagreements shown above and writes a JSON summary with the input hashes.

## 3. Object-local SigLIP 2 features

Native RGB frames, target-lineage masks, and original temporal indices define each history. At each visible target frame, the union of all available lineage masks defines the object support. The crop uses 20% padding and a minimum of four pixels at image boundaries. Pixels outside the support are set to 127 in every RGB channel.

The pinned SigLIP 2 base-patch16-224 revision produces unit-normalized 768-dimensional visual and text embeddings. For description `r`, the default text feature encodes `r`, `a photo of r`, and `the r`, averages the three unit vectors, and normalizes the result.

`configs/siglip2_encoder.json` records the revision, required files, SHA-256 values, dimensions, and Transformers version. Extraction stops if the input file exposes framewise state labels, references, events, or result fields. The extractor records SHA-256 values for the raw inputs and model files:

```text
python scripts/extract_siglip2_features.py --manifest inputs/source_manifest.json --raw-root inputs/rgb --model-directory models/siglip2 --output-root results/features --device cuda
```

The included features make encoder execution unnecessary when reproducing the numerical results.

## 4. Retrieval

For each state, description embeddings are averaged first. The vocabulary origin is the uniform mean of state means, so unequal paraphrase counts cannot change state weights. Déjà Cue subtracts this origin from each description and unit normalizes the residual. The matched absolute method uses the original query. The trajectory-centered extension subtracts the coordinate-wise median of visible visual features before normalization.

Frame scores are calibrated separately for each query using the visible-frame median and `max(1.4826 * MAD, 1e-3)`. A three-tap filter operates independently inside each maximal run of consecutive observed temporal indices. Missing frames split runs and are never crossed.

Candidate lengths use the fixed 33-length schedule selected on development histories. Within each observed run, prefix sums score every valid inclusive window. The primary statistic divides the interval sum by the square root of its inclusive length. Exact score ties select earliest start and then earliest end. Absolute, trajectory-centered, vocabulary-relative, and vocabulary-relative with trajectory centering share features, calibration, runs, candidates, and tie rules.

## 5. Primary, Extended, and Seven-History Results

```text
python scripts/reproduce_main.py --device cpu
python scripts/reproduce_extended.py --device cpu
python scripts/reproduce_seven_history.py --device cpu
```

The primary command evaluates seven methods over 78 histories, 156 states, and 312 descriptions and reproduces all 2,184 selected windows. The extended command reproduces 4,992 windows across 16 matched conditions covering adaptive peak, prompts, duration normalization, candidate ranking, and reference-duration quartiles.

The seven-history command evaluates 7 histories, 16 states, 32 descriptions, 59 reference episodes, and 8 auxiliary tracks under four coordinate origins. All 128 selected windows are compared with the included reference results.

Temporal IoU uses inclusive intervals. R@1 at threshold `tau` records whether the one selected window reaches tIoU `tau`; Top-1 tIoU records that selected window's overlap.

The VOST evaluation averages descriptions within states, states within histories, histories within duplicate-aware source components, and source components uniformly. Its primary confidence intervals use 10,000 component-bootstrap samples, and paired tests use 100,000 component-level sign assignments with Holm correction inside each defined family.

The seven-history evaluation instead averages descriptions within states, states within histories, and histories uniformly. Recurrent states are credited against their best matching annotated occurrence. The statistical seed is 3407.

## 6. Native learned decoders

The shared five-history development set has 13 states, 26 descriptions, 46 positive episodes, and 86 positive run records. Each record contains a frozen 768-dimensional object-local visual feature, two normalized temporal coordinates, vocabulary-relative target text, complete sibling context, span targets, and saliency targets. All seven evaluation histories remain unseen during setting selection and training.

The package executes Moment-DETR, QD-DETR, EaTR, CG-DETR, UVCOM, TR-DETR, TaskWeave, and Sim-DETR through their native proposal, matching, boundary, and loss paths. Training uses 200 epochs, a configured batch-size upper limit of 20, AdamW learning rate `1e-4`, weight decay `1e-4`, and gradient clipping at `0.1`. Architecture-specific contrastive requirements determine the effective groups under that limit while covering all 86 records in every epoch: Moment-DETR and EaTR use five 17--18-record groups; QD-DETR, CG-DETR, and UVCOM use five 18--20-record groups with six support re-exposures; TR-DETR, TaskWeave, and Sim-DETR use 46 two-record groups with six support re-exposures. Sim-DETR drops its learning rate by 0.1 after epoch 100. TaskWeave evaluates final EMA weights; other models evaluate final model weights. There is no validation or test-set checkpoint selection.

Seeds are 3407, 3408, and 3409. CUDA execution requires `CUBLAS_WORKSPACE_CONFIG=:4096:8`; both learned entry points set it before PyTorch import and reject a conflicting value.

```text
python scripts/train_learned_decoder.py --model-id sim_detr --seed 3407 --device cuda:0 --checkpoint results/sim_detr_3407.pt --summary results/sim_detr_3407_training.json
python scripts/evaluate_learned_decoder.py --model-id sim_detr --seed 3407 --device cuda:0 --checkpoint results/sim_detr_3407.pt --output results/sim_detr_3407_evaluation.json
```

Run every model with seeds 3407, 3408, and 3409 to rebuild all 24 results. `scripts/check_reference_results.py` separately recalculates the included three-seed means, sample standard deviations, history-paired deltas, stratified 10,000-sample intervals, exact sign tests, and both Holm-8 families.

Native decoder source and license files are retained under [`third_party/`](../third_party/).

## 7. Diagnostic experiments

The paper's synthetic study simulates 192-frame sequences with true durations 8, 12, 20, 32, and 48. It crosses three signal means, two AR(1) correlations, grid bases 4-6, grid ratios 1.25, 1.5, and 2.0, and three duration normalizations. Each noise cell uses 400 paired trials; the 270 design cells are summarized with a stratified paired 10,000-resample bootstrap using seed 3407.

```text
python scripts/generate_synthetic_duration.py --output results/synthetic_duration.json --check-reference data/reference/robustness/synthetic_duration.json
```

`deja_cue/experiments/perturbations.py` implements deterministic random and contiguous missingness, exact-time distractor mixing, and exact-time identity switches. Randomness is derived from the replicate seed, condition ID, and opaque history ID. Operations return a new history and a summary of the exact changes and input hashes; input arrays are not mutated. The included tracking result uses seeds 3407-3409 and is recalculated by `scripts/check_robustness_results.py`.

The same command recalculates recurrence AP and recall, five-fold development controls, state permutations, within-run circular shifts, hard-negative margins, three-history boundary agreement and sensitivity, and VOST paraphrase consistency from the stored rows.

The timing harness measures the complete scan with precomputed features, synchronizes CUDA around each repetition, and reports milliseconds per query after warmup:

```text
python scripts/benchmark_scan.py --cohort vost --device cuda --warmup 3 --repetitions 10 --output results/scan_benchmark.json
```

Timing varies by hardware and is not expected to match 15.2 ms exactly.

## 8. Verify Files and Results

```text
python scripts/verify_package.py
python -m pytest -q
```

The verification command checks the complete file list and SHA-256 values, relative paths, file formats, array fields, shapes, finite values, unit norms, original frame order, query and reference alignment, cohort mapping, learned settings, and diagnostic results. Tests cover sampling, independent pass validation, consensus rules, cropping, model file checksums, calibration, run boundaries, scan ties, inclusive temporal IoU, aggregation, native learned batching and objectives, perturbations, synthetic generation, timing summaries, reference statistics, robustness calculations, and end-to-end reproduction results.
