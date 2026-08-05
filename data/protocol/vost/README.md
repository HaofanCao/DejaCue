# VOST Protocol Files

`train.txt` and `validation.txt` are the official VOST sequence lists used by `scripts/select_vost_cohort.py`. The sampling result records their SHA-256 values. The sampler reads sequence names only; it does not read annotation labels, reference intervals, features, or method scores.

Two annotators label every selected frame independently. The consensus tools accept:

- one complete frame-label file from each annotator;
- matching history and frame lists in both files; and
- one resolved decision for every disagreement.

The validation code preserves frames on which the annotators already agree and checks that the resulting 78 retained histories, 22 exclusions, and earliest qualifying events match the final cohort.

The two complete annotation files and resolved consensus are not redistributed. This data package contains the split lists, final cohort information, validation code, and the agreement and annotation-sensitivity summaries used by the documented evaluations.
