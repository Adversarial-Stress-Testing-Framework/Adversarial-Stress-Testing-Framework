# Adversarial Stress-Testing Framework for ML-Based Intrusion Detection

Adversarial attacks on network intrusion detection are usually evaluated in
**feature space**, where the attacker may set any feature to any value. That
produces "attack traffic" containing negative byte counts, fractional protocol
identifiers, and connections that retroactively last longer than they did.

No such traffic can be sent. This framework re-measures IDS robustness under
**domain constraints** - restricting the attacker to changes a real network flow
could actually contain - and reports the difference.

On NSL-KDD, reported vulnerability drops by **21.9x +/- 2.8** on differentiable
models, measured over five random splits.

---

## Findings

Every victim is attacked through its **own substitute model** - the adversary
queries it and fits a differentiable stand-in to its outputs - so all four face
an identical attack and stay directly comparable. Substitutes agree with their
victims on 99.4-99.9% of test flows.

Attack success under PGD, black-box. Mean +/- standard deviation over five
random train/test splits:

| Model | Unconstrained | Constrained | Overstated by |
|---|---|---|---|
| LinearSVC | 97.0 +/- 2.9% | 4.5 +/- 0.6% | **21.9 +/- 2.8x** |
| MLP | 77.7 +/- 11.9% | 4.6 +/- 0.8% | **17.2 +/- 3.8x** |
| RandomForest | 87.8 +/- 6.5% | 61.5 +/- 20.0% | 1.6 +/- 0.5x |
| XGBoost | 63.1 +/- 13.7% | 24.2 +/- 12.8% | 3.3 +/- 2.2x |

**The overstatement claim holds for differentiable models and not for trees.**
RandomForest's constrained evasion ranges from 35.8% to 80.7% across splits and
XGBoost's from 9.0% to 40.7% - so their factors have no stable centre and are
reported as ranges. The cause is not the attack: substitute agreement varies by
half a percentage point across the same splits. It is unexplained.

The unconstrained attack corrupts **40 of 40 features** - taking the union of
negative values, fractional counters and out-of-range rates, every feature holds
at least one impossible value on some sample. Negative byte counts, protocol
identifier `1.128`, a 0-second connection becoming 784 seconds. The constrained
attack corrupts none.

### The protocol reorders the models

Detection rate under realistic attack (constrained, black-box PGD):

| Model | Clean | Under attack | Points lost |
|---|---|---|---|
| RandomForest | 99.80% | 38.40% | **-61.4** |
| XGBoost | 99.90% | 75.73% | **-24.2** |
| MLP | 99.37% | 94.76% | -4.6 |
| LinearSVC | 94.04% | 89.83% | -4.2 |

The models with the best clean accuracy degrade hardest. That ordering holds in
every split; the tree magnitudes do not, swinging by up to 45 points. Selecting
an IDS on clean accuracy, or on unconstrained adversarial benchmarks, picks the
more fragile option.

### Adversarial training holds under an adaptive attacker

Retrained on *constrained* adversarial examples, then re-attacked with a
substitute rebuilt against the hardened model. Single split - the multi-seed run
does not yet cover defenses:

| Model | Undefended | Stale substitute | Adaptive |
|---|---|---|---|
| LinearSVC | 5.0% | 0.7% | **2.9%** |
| RandomForest | 36.5% | 0.0% | **0.0%** |
| XGBoost | 31.5% | 0.1% | **0.2%** |
| MLP | 4.9% | 0.1% | **0.5%** |

The adaptive attacker recovers ground over the stale one, as it should, but the
defense still holds - at negligible cost to clean detection.

## What's in here

```
IDS_project/stress_test/
  constraints.py                 feature taxonomy + projection onto realizable flows
  attacks.py                     FGSM, BIM, PGD, minimal-epsilon search, gradients
  victims.py                     LinearSVC, RandomForest, XGBoost, MLP, substitutes
  defense.py                     feature squeezing, realizability filter, adv. training
  metrics.py                     DR, FPR, evasion rate, robustness score, MPD
  spec.py                        dataset feature specifications
  specs/nsl_kdd.yaml             the NSL-KDD taxonomy
  specs/cicids2017.yaml          CICIDS2017 taxonomy (draft, unvalidated)
  run_full_matrix.py             4 models x 3 attacks x 3 defenses, both routes
  run_constrained_comparison.py  attack-budget sweep
  run_multiseed.py               headline metrics over five seeds
  run_tree_variance.py           diagnosis of the tree instability
  run_cicids_audit.py            the same audit on CICIDS2017
IDS_project/dashboard.py         six-tab audit dashboard
EXPERT_QNA_PREP.md               glossary, threat model, known limitations
```

Tree models expose no usable input gradient. Rather than attacking them by
transfer while attacking the differentiable models directly - which confounds
model robustness with attack strength - every victim is routed through its own
substitute. White-box results are reported separately where available.

## Running it

```bash
pip install -r IDS_project/requirements.txt
cd IDS_project
python -m stress_test.run_full_matrix
python -m stress_test.run_constrained_comparison
streamlit run dashboard.py
```

Both runners must be re-run after any change to `constraints.py`, or the
dashboard will show figures from two different versions of the constraint set.

`IDS_project/IDS_project/` holds the earlier RandomForest + Streamlit demo. Run
`python main.py` there first to regenerate `model.pkl` / `scaler.pkl`.

## Constraint model

Each of the 40 surviving NSL-KDD features is classified by what an attacker can
genuinely do to it:

| Role | Count | Rationale |
|---|---|---|
| **Immutable** | 15 | Protocol, service, connection outcome flags. Changing them changes the attack itself, or they are victim-controlled |
| **Increase-only** | 10 | Bytes, duration, connection counts. An attacker can pad and stall; they cannot un-send data |
| **Derived rate** | 15 | Window ratios, bounded and jointly constrained |

Bounds are **learned from training data, not asserted**. An earlier version
hardcoded `same_srv_rate + diff_srv_rate <= 1` on the grounds that it must
logically hold - 3,574 genuine NSL-KDD flows violate it, reaching 1.5, because
the KDD extractor computes the two rates over different windows. That rule
rejected 5.2% of legitimate traffic and tripled the false-positive rate.

## Adding a dataset

The taxonomy lives in a specification file, not in the engine, so a new dataset
is a YAML file rather than a code change:

```yaml
name: my-dataset
features:
  protocol:      {role: immutable}
  bytes_sent:    {role: increase_only, integer: true}
  same_srv_rate: {role: derived_rate}
  packet_len_mean: {role: derived}
ordering:
  - [packet_len_min, packet_len_mean, packet_len_max]
```

Roles are `immutable` (the attacker cannot touch it), `increase_only` (they can
raise but not lower it), `derived_rate` (a ratio in [0, 1]) and `derived` (a
statistic bounded by what real traffic exhibits). Ordering chains declare that
each element must stay <= the next.

Numeric bounds are deliberately absent: they are learned from the training
split. We previously asserted `same_srv_rate + diff_srv_rate <= 1` because it
must logically hold, and 3,574 genuine flows reach 1.5. Specifications declare
structure; the data supplies the numbers.

`FeatureSpec.validate()` reports columns present in the data but missing from
the spec. Those default to immutable, which hands the attacker less than they
have and flatters the results, so they are surfaced rather than ignored.

## Limitations

- **Feature space with problem-space constraints**, not true problem space. No
  packets are generated, and we do not verify a perturbed flow still executes
  the original attack.
- **NSL-KDD only.** 2009 data derived from a 1999 capture. The overstatement
  factor is dataset-specific; the phenomenon should not be.
- **80/20 split of KDDTrain+**, not the standard KDDTest+ protocol, which
  deliberately contains attack types absent from training. Clean accuracy is
  therefore optimistic.
- **Defense figures are single-split.** Attack figures carry error bars over
  five seeds; the defense evaluation does not yet.
- **Tree variance is unexplained.** RandomForest's constrained evasion swings 45
  points across splits. Neither substitute fidelity nor the share of decision
  weight on movable features accounts for it; resolving it needs 20+ seeds.
- The constraint taxonomy is a judgement call. If an attacker can manipulate a
  feature marked immutable, the constrained figures are optimistic.

## Related work

Constrained adversarial attacks on network IDS are an established line of work.
See Pierazzi et al. (IEEE S&P 2020) on problem-space attacks, Sheatsley et al.
on domain constraints, Chernikova & Oprea (FENCE), and Apruzzese et al. on
realistic attacks against NIDS. The contribution here is the measurement
comparison, the like-for-like robustness ranking, and the packaging as a
reusable audit - not the idea of constraining attacks.

## Authors

Jeet Jain, Ishan Dubey, Jeet Vasani
Guided by Dr. Vivek Bhartiya and Abhijeet Jadhav
Thakur College of Engineering and Technology, Mumbai
