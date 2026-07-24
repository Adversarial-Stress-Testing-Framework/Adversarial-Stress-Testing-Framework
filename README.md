# Adversarial Stress-Testing Framework for ML-Based IDS

A pipeline for evaluating the robustness of machine learning-based
intrusion detection systems (IDS) against adversarial evasion attacks,
and for catching defenses that appear effective under naive metrics
but provide no real robustness gain.

Repo: https://github.com/Adversarial-Stress-Testing-Framework/Adversarial-Stress-Testing-Framework

## Current status

Phase 1 complete: LinearSVC baseline, FGSM attack, feature-squeezing
defense evaluation, on NSL-KDD. Extension to additional model families
(Random Forest, CNN) and attack methods (PGD, DeepFool) is in progress
— see Roadmap below.

## Results (Phase 1)

| Metric | Baseline | Under FGSM |
|---|---|---|
| Detection Rate | 94.1% | 1.7% |
| Evasion Rate | — | 98.2% |

**Defense evaluation (feature squeezing):**

| Metric | No defense | With defense |
|---|---|---|
| Raw accuracy | 52.5% | 54.3% |
| Balanced accuracy (flagged-attack rate) | 45.6% | 0.9% |

Raw accuracy suggested the defense helped. Balanced accuracy shows
it didn't — the model mostly stopped predicting "attack" at all.
This is the core finding of Phase 1: naive accuracy is not a
sufficient metric for evaluating IDS defenses under adversarial
conditions.

## Pipeline

1. **Preprocessing** — NSL-KDD cleaning, normalization, feature selection
2. **Model training** — baseline IDS classifier
3. **Attack generation** — adversarial sample crafting
4. **Stress testing** — evaluate model under attack
5. **Defense evaluation** — apply defense, re-evaluate with balanced metrics

## Setup

\`\`\`bash
pip install -r requirements.txt
python run_pipeline.py --model linearsvc --attack fgsm
\`\`\`

## Roadmap

- [ ] Random Forest + PGD
- [ ] CNN + DeepFool
- [ ] Adversarial training as defense
- [ ] Extend to Transformer-based IDS (longer-term)

## Notes

This is ongoing research supporting an IEEE submission. Findings
above are reproducible from this repo; extensions are being added
incrementally rather than held until complete.
