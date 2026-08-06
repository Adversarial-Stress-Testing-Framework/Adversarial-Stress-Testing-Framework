# Expert Q&A — prep sheet

**Read the glossary first.** Getting caught by a word is the only thing that actually
looks bad. Everything else is defensible.

---

## Your opening

> "Adversarial attacks on intrusion detection are usually evaluated in feature space,
> where the attacker can set any feature to any value. We add problem-space domain
> constraints — what an attacker can actually do to a network flow — and re-measure.
> Reported vulnerability drops by about 19x."

Saying **"feature space vs problem space"** in the first 30 seconds signals you've read
the literature. Highest-value phrase available to you.

---

# GLOSSARY — know these cold

| Term | What it means | Where you stand |
|---|---|---|
| **Feature space** | Attacking the numeric vector directly. Can produce impossible values. | What everyone else does |
| **Problem space** | Attacking the real object (actual packets) so the result is genuinely producible. | Our direction — we're partway |
| **Threat model** | The assumed attacker: what they know, what they can touch. | White-box for SVC/MLP, black-box for trees |
| **White-box** | Attacker has the model and its gradients. | LinearSVC, MLP |
| **Black-box** | Attacker only sees inputs/outputs. | RandomForest, XGBoost |
| **Evasion** | Fooling a *deployed* model at test time. | **This is us** |
| **Poisoning** | Corrupting the *training* data. | Not us — say so |
| **Model extraction** | Stealing the model by querying it. | Not us |
| **Surrogate / substitute** | A stand-in model you *can* attack, used to attack one you can't. | How we reach the trees |
| **Transferability** | Adversarial examples built on one model often fool another. | Why the surrogate works |
| **Adaptive attacker** | An attacker who knows your defense and adjusts. | Partially done — see Q6 |
| **Gradient masking** | A "defense" that just hides gradients without adding robustness. Fake security. | The trap we check for |
| **BPDA** | Trick for attacking through non-differentiable defenses. Athalye 2018. | How you'd properly attack feature squeezing |
| **Epsilon (ε)** | Attack budget — how far you may move the input. | Ours = 0.3 std-devs |
| **L∞ / L2** | How you measure "how far". L∞ = biggest single change; L2 = overall distance. | We use L∞ |
| **FGSM** | One big step along the gradient. Fast, weak. | Attack 1 |
| **BIM** | Many small steps. Stronger. | Attack 2 |
| **PGD** | BIM + random start + restarts. Strongest standard attack. | Attack 3 |
| **C&W** | Optimization attack, finds *minimum* perturbation. Slow, very strong. | Not implemented — admit it |
| **Adversarial training** | Train on adversarial examples so the model learns them. | Our working defense |
| **Certified robustness** | Mathematical *proof* of robustness, not just empirical. | Not done. Out of scope |
| **Base rate fallacy** | Low FPR still means huge alert volume at real traffic scale. | See Q11 — expect this |
| **Concept drift** | Traffic changes over time; the model goes stale. | Not addressed |

---

# THE QUESTIONS

## 1. "Feature space or problem space?"

*Pierazzi et al., IEEE S&P 2020 — "Intriguing Properties of Adversarial ML in the
Problem Space."* **The** question in this field.

> "Feature space, but with problem-space constraints projected back onto it. We're not
> claiming full problem-space — we don't generate packets. We enforce what a real flow
> must satisfy: immutable protocol/service fields, monotonic byte and duration counters,
> valid rate ranges. Generating actual PCAPs and re-extracting features is the next step
> and we haven't done it."

Do not overclaim. Admitting the gap is what makes everything else credible.

## 2. "NSL-KDD? That's 2009, from 1999 traffic."

They're right. Don't argue.

> "Agreed, and we say so in the paper. We use it because it's the standard benchmark and
> keeps us comparable to prior work. The contribution isn't the dataset — it's the
> constraint specification, which is dataset-agnostic. Porting to CICIDS2017 means
> writing a new spec, not rewriting the method."

## 3. "You split KDDTrain+ 80/20? The protocol uses KDDTest+."

**Our weakest methodological point. Know it before they say it.**

KDDTest+ deliberately contains attack types absent from training. Splitting the training
file means train and test share a distribution — which is why we see 99.9% where
published NSL-KDD results sit at 75–80%.

> "Correct, and it inflates our clean accuracy. We measure *relative degradation* under
> attack rather than absolute detection, so the comparison holds — but the absolute
> numbers are optimistic. Moving to KDDTest+ is queued."

## 4. "How does an attacker actually change `same_srv_rate`?"

> "They don't write it directly. It's a ratio over a 2-second window, so the attacker
> influences it by choosing which connections to open. That's why we classified rates as
> indirectly influenceable — bounded and jointly constrained, but movable — rather than
> freely writable."

## 5. "What's your threat model?"

> "Two. White-box for LinearSVC and the MLP — attacker has the model and gradients.
> Black-box transfer for RandomForest and XGBoost, since trees expose no input gradient:
> the attack is built on a surrogate and transferred. We report them separately because
> they're not comparable."

## 6. "Have you tested an adaptive attacker?"

Be precise — this is where people overclaim.

> "For the differentiable models, yes — after adversarial training we re-attack using the
> hardened model's own gradients. For the tree models, not yet: we re-attacked with the
> original surrogate rather than rebuilding it against the hardened model, so those
> numbers are optimistic. The substitute-retraining version is implemented but I haven't
> run it yet. It's our top open item."

## 7. "Why feature squeezing? Athalye broke that in 2018."

> "We know — it's a negative control, not a proposed defense. Our own results show it
> fails: it reduces evasion by collapsing predictions toward the majority class, not by
> recovering detection. Adversarial training is the defense we actually report."

## 8. "Does your constraint set guarantee realizability?"

No. Be honest.

> "Necessary, not sufficient. We enforce per-feature rules and pairwise rate consistency.
> We don't enforce full joint consistency across all 41 features, and we don't verify the
> perturbed flow still executes the original attack."

## 9. "Does the attack still *work* after you perturb it?"

The deepest question. Honest answer: we assume, we don't verify.

> "We assume padding bytes and adding delay preserve function — which is exactly why
> those are the features we allow to move. We haven't verified end-to-end. That needs the
> PCAP round-trip and it's the biggest thing we haven't done."

## 10. "What does ε = 0.3 mean physically?"

> "Currently nothing — it's 0.3 standard deviations in scaled space, which means nothing
> to an operator. Converting the budget to real units — bytes of padding, seconds of
> delay — is on the roadmap. Fair criticism."

## 11. "What's your false positive rate at real traffic volume?" ⚠️

**The practitioner's first instinct. Have the arithmetic ready.**

At ~1M flows/day, roughly 900k benign:

| Model | FPR | False alarms/day |
|---|---|---|
| LinearSVC | 3.29% | **~29,600** |
| MLP | 0.48% | ~4,300 |
| XGBoost | 0.09% | ~810 |
| RandomForest | 0.04% | **~360** |

> "Our LinearSVC baseline is operationally unusable at scale — roughly 30,000 alerts a
> day. RandomForest at 360/day is borderline for a small SOC. We optimised for
> comparability with prior work, not deployability. And adversarial training pushes
> LinearSVC's FPR from 3.29% to 4.37%, so the defense isn't free at scale either."

Volunteering this makes you look like you think about deployment, not just benchmarks.

## 12. "How do you know your defense isn't just gradient masking?"

> "That's why we run PGD with random restarts rather than only FGSM, and why we report
> the transfer attack separately. A masked-gradient defense collapses under transfer, and
> ours doesn't. But we haven't run a full BPDA check on feature squeezing — that would be
> the rigorous test."

## 13. "Why not just use ART or CleverHans or Foolbox?"

> "They're built for images, where every pixel value is valid. None of them model domain
> constraints for network flows — that's precisely the gap we're filling. The attacks
> themselves are standard; the constraint layer is what's new."

## 14. "Binary classification? Real IDS classify attack types."

> "Binary normal/attack. Multi-class per attack category — DoS, Probe, R2L, U2R — would
> be more realistic, and the constraint set is arguably different per category. R2L and
> U2R are also badly under-represented in NSL-KDD. Not addressed."

## 15. "Does this work on encrypted traffic?"

> "Our features are flow-level statistics — bytes, timing, counts — not payload, so in
> principle they survive encryption. But we haven't tested it, and modern encrypted
> traffic analysis uses a different feature set entirely."

## 16. "What about poisoning rather than evasion?"

> "Out of scope. We assume a clean training set and attack at inference time. Poisoning
> is a different threat model and would need a different framework."

## 17. "Real-time performance?"

> "We measure inference latency but the audit itself is offline — it's a pre-deployment
> testing tool, not an inline component. Adversarial training costs nothing at inference
> since it's a training-time change."

---

# NUMBERS TO HAVE READY

| | |
|---|---|
| Overstatement factor | **18.7x** (LinearSVC, FGSM, ε=0.3) |
| Detection under fake test | 94.1% → **1.7%** |
| Detection under honest test | 94.1% → **89.2%** |
| Models | 4 — LinearSVC, RandomForest, XGBoost, MLP |
| Attacks | 3 — FGSM, BIM, PGD |
| Worst realistic case | RandomForest 99.8% → **69.9%** (constrained PGD) |
| Best defense | Adversarial training, RF 30% → 0.03% evasion *(optimistic — see Q6)* |
| Feature split | 15 locked / 10 increase-only / 15 rate |
| Dataset | NSL-KDD KDDTrain+, 126k flows, 80/20 split |

**Lead with this, not the 19x:** the unconstrained test doesn't just inflate numbers — it
*reorders* the models. LinearSVC looks worst and is actually most stable. RandomForest
looks safest at 99.8% and degrades hardest.

---

# THE STORY THAT MAKES YOU LOOK GOOD

> "We asserted `same_srv_rate + diff_srv_rate ≤ 1` because logically it must. Then we
> validated against real NSL-KDD traffic and found 3,574 rows that break it — it reaches
> 1.5, because the KDD extractor computes the two rates over different windows and rounds
> them. Our rule was wrong, not the data. It was rejecting 5.2% of legitimate traffic and
> tripling the false-positive rate. We now learn the bounds from data instead of assuming
> them."

Experts trust people who catch their own mistakes far more than people with clean
stories. This is your best 30 seconds.

---

# ASK THEM THESE

1. "Which flow features can an attacker realistically manipulate in the wild? We may have
   been too conservative locking the outcome fields."
2. "We assume padding bytes and adding delay preserve an exploit's function. Does that
   hold, or does it break more attacks than we think?"
3. "Is ML-IDS evasion something you see in practice, or still mostly a research concern?"
4. "If a tool audited an IDS and returned a security grade — what would need to be on
   that report for it to be useful to you?"
5. "What false-positive rate is actually tolerable in a real SOC?"

Q4 and Q5 are direct requirements-gathering for the capstone tool.

---

# DO NOT SAY

- ~~"We proved IDS are secure."~~ We didn't. 5–30% still evade.
- ~~"Our defense stops all attacks."~~ Tree numbers are optimistic and untested adaptively.
- ~~"Nobody has done this."~~ Say what's new: *constraint-aware auditing*. The attacks and
  defenses are all textbook.
- ~~"Our accuracy is 99.9%."~~ Always attach: *"on an 80/20 split of the training file,
  which inflates it."*
- Anything you can't define. **If they use a term you don't know, ask.** Guessing is the
  only thing that actually looks bad.

---

# IF YOU GET STUCK

> "I don't know — that's not something we've tested. How would you approach it?"

Perfectly good answer. Turns a gap into a conversation, and experts respect it far more
than a bluff.
