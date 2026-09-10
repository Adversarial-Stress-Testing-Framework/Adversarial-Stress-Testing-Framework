# Adversarial Stress-Testing of ML-Based IDS — Results

Every figure below comes from a committed JSON report in `IDS_project/`. Run
`demo.bat` for the interactive dashboard, or the runners listed at the bottom to
regenerate anything here.

Attack figures are **mean ± standard deviation over random train/test splits**
(five on NSL-KDD, three on CICIDS2017). Where a number is a single split, it
says so.

---

## The finding

Adversarial attacks on network intrusion detection are evaluated in *feature
space*, where the attacker may set any feature to any value. Converted back into
network units, what those attacks produce is not traffic:

| Feature | Real value | Attack produced | Why it cannot exist |
|---|---|---|---|
| `src_bytes` | 105 bytes | **−1,614,895** | negative data volume |
| `protocol_type` | 1 (UDP) | **1.128** | there are three protocols, not a range |
| `duration` | 0 seconds | **784 seconds** | the connection was instantaneous |

On NSL-KDD the unconstrained attack corrupts **40 of 40 features** — taking the
union of negative values, fractional counters and out-of-range rates, every
feature holds an impossible value on some sample. On CICIDS2017 it drives **59
of 70 features negative** and violates **all 16** packet-length and
inter-arrival ordering constraints, emitting flows whose smallest packet exceeds
their largest.

The constrained attack produces **zero** violations of any kind on either
dataset.

---

## 1. NSL-KDD — the headline, and its limits

Attack success under black-box PGD, five splits:

| Model | Unconstrained | Constrained | Overstated by |
|---|---|---|---|
| LinearSVC | 97.0 ± 2.9% | 4.5 ± 0.6% | **21.9 ± 2.8×** |
| MLP | 77.7 ± 11.9% | 4.6 ± 0.8% | **17.2 ± 3.8×** |
| RandomForest | 87.8 ± 6.5% | 61.5 ± 20.0% | 1.6 ± 0.5× |
| XGBoost | 63.1 ± 13.7% | 24.2 ± 12.8% | 3.3 ± 2.2× |

**The overstatement claim holds for differentiable models and not for trees.**
RandomForest's constrained evasion ranges 35.8–80.7% across splits and
XGBoost's 9.0–40.7%, so their factors have no stable centre. The cause is not
the attack — substitute agreement varies by half a percentage point over the
same splits — and remains unexplained.

### The evaluation protocol reorders the models

Detection rate under realistic attack:

| Model | Clean | Under attack | Points lost |
|---|---|---|---|
| RandomForest | 99.80% | 38.40% | **−61.4** |
| XGBoost | 99.90% | 75.73% | **−24.2** |
| MLP | 99.37% | 94.76% | −4.6 |
| LinearSVC | 94.04% | 89.83% | −4.2 |

The models with the best clean accuracy degrade hardest. **That ordering holds
in every split.** Selecting an IDS on clean accuracy, or on published
unconstrained benchmarks, picks the more fragile option.

---

## 2. CICIDS2017 — the headline does not generalise

| | NSL-KDD | CICIDS2017 |
|---|---|---|
| Unconstrained evasion | 97.0% | 98.5% |
| Constrained evasion | 4.5% | **94.2%** |
| Overstatement | **21.9×** | **1.05 ± 0.002×** |

Constraining barely helps on CICIDS2017, and the surviving evasions are fully
realizable — zero violations of any kind. That detector genuinely is that
vulnerable to attacks a real adversary could send.

**This separates two claims NSL-KDD had conflated:**

- *Unconstrained evaluation produces physically impossible traffic.* Replicates
  on both datasets. Appears universal.
- *Constraining substantially reduces measured evasion.* Does **not** replicate.

Why: NSL-KDD's 15 rate features are bounded to [0, 1], which boxes an attacker
in. CICIDS2017's 78 columns leave 25 derived and 13 increase-only features
movable — enough room to evade regardless of what is locked.

**How much domain constraints protect a detector depends on how much of its
feature space remains attacker-influenceable.** That is a property of the schema,
not a constant.

---

## 3. Packet-level realizability

Feature-space results assume a vector can be realised as traffic. This checks it,
running forward from 400 real FTP-Patator flows pulled out of the 10.5 GB
Tuesday capture, perturbed with operations an attacker can actually perform.

| Rung | Evasion |
|---|---|
| Unconstrained feature space | 100% |
| Constrained feature space | 100% |
| **Realized in packets** | **0%** |

Zero at every budget tested, up to 1024 bytes of padding and 500 ms of delay.
The perturbations are not inert — that budget moves 30 of 78 features, taking
Average Packet Size from 7.4 to 169.1 — the detector holds anyway.

**So even constrained feature-space evaluation overstates what an attacker
limited to real packet operations achieves.**

⚠️ **Indicative, not definitive.** The extractor is a reimplementation, not
CICFlowMeter — the official tool is Java over `jnetpcap` and impractical to build
here. It agrees with the published FTP-Patator rows on 3 of 9 checked features,
reading high on durations and packet counts. The released CSVs strip IP
addresses, so no extracted flow can be matched to its published counterpart and
the disagreement cannot be resolved directly. The comparison is internally
consistent — the same extractor produces both sides — so the relative
measurement is sounder than the absolute fidelity suggests.

---

## 4. A defense that holds

Retrained on constrained adversarial examples, then re-attacked with a substitute
rebuilt against the hardened model. **NSL-KDD, single split** — the multi-seed
run does not cover defenses.

| Model | Undefended | Adaptive attacker |
|---|---|---|
| LinearSVC | 5.0% | **2.9%** |
| RandomForest | 36.5% | **0.0%** |
| XGBoost | 31.5% | **0.2%** |
| MLP | 4.9% | **0.5%** |

At no measurable cost to clean detection. Feature squeezing, by contrast, does
not work: it reduces evasion by refusing to predict "attack", not by detecting
better.

---

## 5. Mistakes caught in our own work

Recorded because each one changed a number we were reporting.

- **A rule we asserted was false.** `same_srv_rate + diff_srv_rate <= 1` must
  logically hold; 3,574 genuine NSL-KDD flows reach 1.5. It was rejecting 5.2% of
  legitimate traffic and tripling the false-positive rate. Bounds are now learned
  from data, never asserted.
- **The model comparison was confounded.** Differentiable victims were attacked
  white-box while trees could only be reached by transfer — a strong attack
  against a weak one, called robustness. Every victim now goes through its own
  substitute. The ranking inversion survived the correction.
- **A metric carried no information.** The reported mean perturbation distance
  was mathematically the constant ε√d.
- **The violation counter underreported twice.** It checked non-negativity only
  on integer features, reporting 0 negative-valued features on CICIDS where the
  truth was 59; and it compared against exact zero, flagging a float round-trip
  of 1.0000000000000002 as an out-of-range rate.
- **`23 of 40` was really `40 of 40`.** The original count included only
  fractional integers.

---

## Limitations

- **Feature space with problem-space constraints.** Packets are perturbed and
  re-extracted, but with a reimplemented extractor that does not fully match
  CICFlowMeter, and we never verify a perturbed flow still executes its attack.
- **Defense figures are single-split.** Attack figures carry error bars.
- **Tree variance is unexplained.** RandomForest swings 45 points across splits;
  neither substitute fidelity nor movable-feature importance accounts for it.
  Resolving it needs 20+ seeds.
- **Optimistic clean accuracy on NSL-KDD.** An 80/20 split of `KDDTrain+` rather
  than the standard `KDDTest+` protocol, which withholds unseen attack types.
- **The taxonomy is a judgement call.** If an attacker can move a feature marked
  immutable, the constrained figures are optimistic.

---

## Reproducing

```bash
cd IDS_project
python -m stress_test.run_constrained_comparison   # NSL-KDD budget sweep
python -m stress_test.run_full_matrix              # 4 models x 3 attacks x 3 defenses
python -m stress_test.run_multiseed                # error bars over five splits
python -m stress_test.run_cicids_audit             # CICIDS2017 (needs the CSVs)
python -m stress_test.run_packet_roundtrip         # packet-level (needs the PCAP)
streamlit run dashboard.py
```

Datasets are not tracked; see `data/cicids2017/README.md`. Adding a dataset is a
YAML specification file, not a code change — the format is documented in the
main `README.md`.
