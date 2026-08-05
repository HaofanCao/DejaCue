# Installation and Environment

## Before You Install

Clone the complete repository before installing. The commands read the included `data/`, `configs/`, `scripts/`, and `third_party/` directories, so keep them together and use the editable installation shown below. A wheel built from `pyproject.toml` contains the Python modules but not the full data and reproduction files.

## Complete installation

Create an environment with Python 3.10 or newer from the repository root:

```bash
python -m venv .venv
```

Activate it on POSIX systems:

```bash
source .venv/bin/activate
```

Or activate it from Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install the project:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[preprocess,test]"
```

The editable install includes core evaluation, learned-decoder, SigLIP 2 preprocessing, and test dependencies. Alternatively, invoke the environment's Python executable directly (`.venv/bin/python` on POSIX and `.venv\Scripts\python.exe` on Windows).

The declared ranges and pins are:

| Component                      | Supported version |
| ------------------------------ | ----------------- |
| Python                         | `>=3.10`          |
| NumPy                          | `>=1.26,<3`       |
| PyTorch                        | `>=2.2,<3`        |
| torchvision                    | `>=0.17,<1`       |
| SciPy                          | `>=1.11,<2`       |
| scikit-learn                   | `>=1.4,<2`        |
| einops                         | `>=0.7,<1`        |
| pytest                         | `>=8,<10`         |
| Transformers for extraction    | `4.57.6`          |
| tokenizers for extraction      | `0.22.2`          |
| safetensors for extraction     | `0.8.0`           |
| huggingface-hub for extraction | `0.36.2`          |
| Pillow for extraction          | `>=10,<13`        |

`pyproject.toml` declares the dependencies. `requirements.txt` provides a short installation command for reproduction and testing.

## Reproduce Results

CPU is sufficient for file verification, statistical calculations, all three fixed-feature evaluations, and tests:

```text
python scripts/reproduce_all.py --device cpu
```

CUDA reproduction is optional:

```text
python scripts/reproduce_all.py --device cuda
```

No compiler or custom extension is required. Exact selected intervals must match; floating-point comparisons use the tolerances implemented by the evaluation and verification code. Visual feature rows are loaded into float32 computation and validated with unit-norm tolerance `2e-3`.

## SigLIP 2 feature extraction

Extraction requires the `preprocess` extra, raw RGB and mask inputs, and a local model directory. `configs/siglip2_encoder.json` pins the exact model revision, seven required files, their sizes and SHA-256 values, the 768-dimensional output, and Transformers 4.57.6. The loader rejects a version mismatch, missing file, size mismatch, or hash mismatch before model construction.

CUDA is recommended for full extraction, but the entry point also accepts CPU:

```text
python scripts/extract_siglip2_features.py --manifest inputs/source_manifest.json --raw-root inputs/rgb --model-directory models/siglip2 --output-root results/features --device cuda --batch-size 64
```

The extractor writes to a new output directory and records relative paths and input hashes. It does not fetch model or raw-data files.

## Deterministic learned training

CUDA training must use:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

`scripts/train_learned_decoder.py` and `scripts/evaluate_learned_decoder.py` set this value before importing PyTorch. The shared protocol checks it again and rejects a conflicting pre-existing value. The same entry points seed Python, NumPy, CPU PyTorch, and every CUDA device; enable deterministic algorithms; disable cuDNN benchmarking; and enable deterministic cuDNN behavior.

Use an explicit CUDA device for the 200-epoch runs:

```text
python scripts/train_learned_decoder.py --model-id sim_detr --seed 3407 --device cuda:0 --checkpoint results/sim_detr_3407.pt --summary results/sim_detr_3407_training.json
python scripts/evaluate_learned_decoder.py --model-id sim_detr --seed 3407 --device cuda:0 --checkpoint results/sim_detr_3407.pt --output results/sim_detr_3407_evaluation.json
```

Outputs must not already exist. Each checkpoint stores the model ID, seed, training configuration, architecture information, and parameter SHA-256. Evaluation stops if its saved model, seed, or training settings differ from the current command.

Deterministic execution controls algorithm selection, not hardware equivalence. For the closest comparisons, keep the Python package set, PyTorch build, CUDA runtime, accelerator architecture, and driver fixed across runs.

## Timing benchmark

The timing harness synchronizes CUDA before and after each measured repetition, warms up separately, and reports the median and interquartile range in milliseconds per query:

```text
python scripts/benchmark_scan.py --cohort vost --device cuda --warmup 3 --repetitions 10 --output results/scan_benchmark.json
```

The paper reports 15.2 ms per query with precomputed features on an RTX 4090. New measurements are hardware-specific and are not required to equal that value. The generated JSON records only the device class, workload size, schedule size, warmup count, repetition count, and timing summary.
