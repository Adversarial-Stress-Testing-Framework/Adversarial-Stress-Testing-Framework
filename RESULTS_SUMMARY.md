# 🛡️ Adversarial Stress-Test Audit Dashboard Summary

> **Note:** This is a static summary of the core results extracted from the framework's pre-generated JSON experiment reports. To view the interactive dashboard, run `demo.bat` in the project root.

## 🏆 Headline Findings (NSL-KDD, LinearSVC)

| Metric | Unconstrained | Constrained | Impact |
|---|---|---|---|
| **Attack Wins (Evasion Rate)** | 98.2% | 5.3% | Vulnerability **Overstated by 18.7×** |
| **Never Evadable** | - | 2.0% | Attacks blocked permanently by locked features |

---

## 📊 Clean Performance vs. Realistic Attack

How the 4 victim models perform under normal conditions vs. a realistic (constrained, black-box PGD) attack.

| Model | Clean Detection Rate | Under Attack | Points Lost |
|---|---|---|---|
| **RandomForest** | 99.8% | 63.4% | -36.4 |
| **XGBoost** | 99.9% | 68.5% | -31.4 |
| **LinearSVC** | 94.1% | 89.5% | -4.6 |
| **MLP** | 99.5% | 94.6% | -4.9 |

> **Ranking Inversion:** The models that look safest under clean testing (RandomForest, XGBoost) are actually the most fragile under a realistic attack. The ordering flips when constraints are applied.

---

## ⚔️ Unconstrained vs. Constrained Attacks

*Black-box PGD attack evasion rates across models.*

| Model | Unconstrained Evasion | Constrained Evasion | Overstatement Factor |
|---|---|---|---|
| **LinearSVC** | 98.2% | 5.0% | **19.6×** |
| **RandomForest** | 78.6% | 36.5% | **2.2×** |
| **XGBoost** | 73.1% | 31.5% | **2.3×** |
| **MLP** | 91.6% | 4.9% | **18.7×** |

---

## 🛡️ Defense Evaluation

Evasion rate against a constrained PGD attack, comparing no defense against standard mitigation strategies.

| Model | No Defense | Feature Squeezing | Realizability Filter | Adversarial Training (Adaptive) |
|---|---|---|---|---|
| **LinearSVC** | 5.0% | 3.2% | 5.0% | **2.9%** |
| **RandomForest** | 36.5% | 5.1% | 36.5% | **0.0%** |
| **XGBoost** | 31.5% | 4.7% | 31.5% | **0.2%** |
| **MLP** | 4.9% | 2.9% | 4.9% | **0.5%** |

> **Realizability Filter:** The filter eliminates unconstrained attacks completely but does **nothing** against constrained ones. It forces the attacker to play by the rules.
> 
> **Adversarial Training:** Retraining on constrained adversarial examples effectively hardens the models against adaptive attackers at almost zero cost to clean detection.

---

## 📉 Attack Budget Sweep (ε)

As the attacker's budget grows, the gap between constrained and unconstrained attacks closes. At small budgets (the realistic regime), unconstrained evaluation is highly misleading.

| Budget (ε) | Constrained Evasion | Unconstrained Evasion | Overstatement |
|---|---|---|---|
| **0.1** | 1.8% | 38.5% | **21.1×** |
| **0.3** | 5.3% | 98.2% | **18.7×** |
| **0.5** | 12.1% | 99.7% | **8.2×** |
| **1.0** | 69.9% | 99.9% | **1.4×** |
| **5.0** | 98.0% | 100.0% | **1.0×** |

---

## 🌐 Dataset Generalization: CICIDS2017

Testing on the more complex CICIDS2017 dataset (78 features, many computed/derived).

| Metric | Result (Average over 3 seeds) |
|---|---|
| **Clean Detection Rate** | 80.5% |
| **Unconstrained Evasion** | 98.5% |
| **Constrained Evasion** | 94.2% |
| **Overstatement Factor** | **1.05×** |

> The overstatement factor is only **1.05×** on CICIDS2017. Because the model's clean detection rate is relatively low (80.5%), it is already fundamentally fragile on this dataset. Constraints don't protect a model that struggles to detect clean attacks in the first place.
