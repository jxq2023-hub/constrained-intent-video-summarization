# Reproduction guide

## Level 1: release integrity (no third-party data required)

From the repository root:

```bash
python code/reproduce/verify_release.py
python code/reproduce/summarize_release.py
```

The first command checks expected record counts, pseudonymisation, forbidden file types, and selected-index parseability. The second prints compact means from the released main, ablation, query-swap, and human-rating tables.

## Level 2: re-analysis of released records

The original analysis scripts are in `code/research_q3/experiments/`. They document the statistical procedures used to generate the accompanying summary tables. Paths in those scripts reflect the original project layout; copy the relevant released CSV into `code/research_q3/results/` before running an original analysis entry point.

## Level 3: full inference

Full inference requires authorised QFVS/TVSum/SumMe inputs and pretrained weights, which are not redistributable through this package. Follow `THIRD_PARTY_DATASETS.md`, place local files at the expected paths, and run from `code/`:

```bash
python -m research_q3.experiments.run_qfvs_protocol --help
python -m research_q3.experiments.run_qfvs_nested_selection
python -m research_q3.experiments.run_qfvs_end_to_end_intent
python -m research_q3.experiments.run_qfvs_simulated_user_language --help
python -m research_q3.experiments.run_standard_protocol --help
```

The reported local language model was Qwen2.5:7b served through Ollama's OpenAI-compatible endpoint. No external API key is required.
