# Data and Feature Terms

The `data/` directory combines metadata and reference results created for Déjà Cue with features derived from external research datasets and a frozen vision-language encoder. These materials do not share one blanket software or dataset license.

## What Is Included

The dataset contains frozen embeddings, stable evaluation IDs, fixed method settings, reference intervals, statistical results, and validation files. It does not contain native RGB or video, target-lineage masks, pretrained encoder weights, complete annotation files, resolved framewise consensus, or generated decoder checkpoints.

A derived feature does not transfer ownership of, or grant new rights to, its source media or model. Users remain responsible for the terms of every upstream asset they access or redistribute.

## Source Terms

| Asset family | Upstream source | Files in this dataset | Terms to retain or verify |
| --- | --- | --- | --- |
| VOST histories | [VOST dataset](https://www.vostdataset.org) and [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Tokmakov_Breaking_the_Object_in_Video_Object_Segmentation_CVPR_2023_paper.html) | Object-local embeddings, split lists, evaluation IDs, and reference intervals | VOST identifies [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); also follow the underlying source-media terms identified by VOST |
| HyperNeRF histories | [HyperNeRF](https://doi.org/10.1145/3478513.3480487) | Target and auxiliary-track embeddings plus evaluation information | Verify and retain the upstream project and dataset terms for the scenes used |
| Neural 3D Video history | [Neural 3D Video](https://doi.org/10.1109/CVPR52688.2022.00544) | Coffee-martini target embeddings and evaluation information | Verify and retain the upstream project and dataset terms |
| D-NeRF recurrence experiment | [D-NeRF](https://doi.org/10.1109/CVPR46437.2021.01018) | Predictions and reference intervals; no rendered RGB | Verify and retain the upstream project and dataset terms |
| Feature encoder | [SigLIP 2](https://arxiv.org/abs/2502.14786) and its [model card](https://huggingface.co/google/siglip2-base-patch16-224) | Unit-normalized embeddings; no model files | Follow the model-card and weight terms for the specified revision |

VOST documentation requests citation of VOST, Ego4D, and EPIC-KITCHENS when its sequences are used. Retain those attributions whenever the corresponding VOST-derived files are redistributed.

Feature extraction uses `google/siglip2-base-patch16-224` at revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`. The checksum list identifies the model files used for extraction, but those model files are not included.

## Material Created for Déjà Cue

The JSON metadata, file formats, protocol descriptions, and reference results created for Déjà Cue may be reused for research and method comparison with attribution to **Déjà Cue: Localizing States in Object Histories via Vocabulary-Relative Coordinates**. The Apache-2.0 license in the companion code repository applies to original software; it does not automatically cover external data, embeddings, or upstream media.

## Redistribution

When redistributing these files or a modified version:

1. retain this file and the applicable upstream notices;
2. cite Déjà Cue and the upstream sources whose assets were used;
3. document every added, removed, or transformed file;
4. preserve the stable evaluation IDs, or provide an explicit mapping when they change; and
5. regenerate the checksum list for the redistributed files.

Do not describe the package as granting rights to source media, model weights, or personal likenesses beyond those supplied by their owners.

## Privacy and Sensitive Use

The dataset omits direct identifiers and native media, but frozen embeddings are derived from external research content and should not be treated as anonymous by default. These files are intended for scientific retrieval and reproducibility research, not biometric identification, surveillance, or sensitive-attribute inference.

These terms do not replace the upstream licenses or terms of use.
