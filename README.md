# Constrained-intent query-focused video summarization

This repository package contains the compact code and derived data needed to audit the experiments reported in the accompanying manuscript. It does **not** redistribute QFVS, TVSum, or SumMe videos, third-party annotations, pretrained model weights, signed consent forms, identity mappings, or credentials.

## Contents

- `code/`: constrained intent parsing, visual/semantic scoring, selection, evaluation, and analysis scripts.
- `data/query_requests/`: 1,080 researcher-constructed Chinese natural-language stress-test requests. These are controlled test inputs, not observations from real users.
- `data/intent_outputs/`: structured intent outputs and end-to-end routing results.
- `data/concept_mappings/`: QFVS query index and Chinese/English concept mapping.
- `data/splits/`: fixed leave-one-video-out QFVS splits and video identifiers used for TVSum/SumMe.
- `data/selected_shots/`: selected QFVS shot indices and method-level metrics.
- `data/evaluation_results/`: main quantitative comparison tables.
- `data/query_swap/`: matched-query versus mismatched-query discrimination results.
- `data/ablation/`: nested selection, parser/routing, language-stress, and fusion-sensitivity results.
- `data/human_evaluation/`: pseudonymous five-rater scores and derived statistics.
- `docs/`: third-party dataset acquisition, human-rating protocol, data dictionary, and reproduction notes.

## Quick verification

Python 3.9 was used for the reported experiments. A lightweight integrity check does not require the original videos:

```bash
python code/reproduce/verify_release.py
python code/reproduce/summarize_release.py
```

Full model inference requires authorised local copies of the third-party datasets and the model weights described in `docs/THIRD_PARTY_DATASETS.md`. Paths in `code/config.py` are resolved relative to the repository root; cloud credentials are not required because the reported intent parser used a local Ollama-compatible endpoint.

## Data and licensing boundary

Code is released under the MIT License. Author-generated derived records under `data/` are released under CC BY 4.0, except where a file contains identifiers or transformations tied to a third-party benchmark; those records remain subject to the source dataset's terms. No ownership claim is made over QFVS, TVSum, or SumMe.

## Human evaluation

Five adults voluntarily completed the blind rating task after receiving study information and providing informed consent. Public files contain only pseudonymous rater identifiers and numerical ratings; names, contact details, signed forms, and the private randomisation key are excluded. Institutional ethics approval or exemption has not been represented here because no documented determination was supplied for this release.

## Citation and versioning

After upload, create a fixed GitHub release tagged `v1.0.0` and cite both the release URL and its full commit hash. Update the placeholder repository URL in `CITATION.cff` only after the public repository address is known.
