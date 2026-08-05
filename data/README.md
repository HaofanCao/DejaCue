# Data Layout

This directory contains the fixed features and evaluation files used by the repository commands.

| Path | Role |
| --- | --- |
| `benchmark.json` | Primary 78-history benchmark manifest and file references |
| `features/` | VOST visual, text, and prompt-variant SigLIP 2 features |
| `learned/development/` | Five-history inputs for native decoder training |
| `seven_history/` | Seven-history evaluation features and auxiliary tracks |
| `reference/` | Exact windows and the rows used for the reported calculations |
| `protocol/vost/` | Fixed VOST split lists and protocol notes |

Paths inside manifests are repository-relative. Loaders reject absolute paths, parent traversal, unexpected array fields, non-finite values, misaligned rows, and invalid feature norms. See the repository-level [`README.md`](../README.md) for the release overview and [`DATA_LICENSE.md`](../DATA_LICENSE.md) for terms.
