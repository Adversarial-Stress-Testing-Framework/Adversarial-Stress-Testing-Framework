# Running the project live

Everything below was timed on this machine. Nothing here is estimated.

---

## The 30-second answer

```bash
demo.bat
```

Double-click it, or run it from the repo root. It activates the virtual
environment and opens the audit dashboard at <http://localhost:8501>.

**It needs no datasets.** All seven report JSONs are committed, so the
dashboard renders every result on a fresh clone with an empty `data/`
folder. Takes about 20 seconds to come up the first time.

That is the whole demo if you only have two minutes.

---

## The five-minute walkthrough

Open `demo.bat`, then move left to right across the tabs.

| # | Tab | What to point at | What to say |
|---|---|---|---|
| 1 | **Overview** | The four metric tiles | "97% attack success drops to 4.5% when the attacker is held to traffic that can physically exist. That's 21.9×, averaged over five random splits." |
| 2 | **Overview**, lower | Three rungs of realism | "100% in feature space, 0% once the same attacker is restricted to real packet operations." |
| 3 | **Constraints** | Violation chart | "The unconstrained attack produces negative byte counts and fractional protocol IDs. The constrained one produces none." |
| 4 | **Error bars** | The four facets + Stable? column | "Every figure is a mean over five splits. The tree models say NO in that column — their factors have no stable centre, and we report ranges." |
| 5 | **Second dataset** | 21.9× beside 1.05× | "Our headline did not generalise. That's the finding: the effect depends on the schema, not on the attack." |
| 6 | **Packet round-trip** | 3 of 9 extractor table | "Our extractor is a reimplementation. It agrees on 3 of 9 features. We call this rung indicative, not definitive." |

Slide 5 is the one to land. Leading with the result that broke your own
claim is what makes the rest credible.

---

## If they ask "is this real, or just saved numbers?"

Run one live. **Open a second terminal — do not stop the dashboard.**

```bash
cd IDS_project
..\.venv\Scripts\python.exe -m stress_test.run_constrained_comparison
```

**45 seconds.** Prints the budget sweep and ends with:

```
    unconstrained evasion rate : 0.9820
    constrained evasion rate   : 0.0526  (best of FGSM / PGD)
    overstatement factor       : 18.66x  at eps=0.3
```

Then refresh the dashboard — it keys its cache on file modification time, so
the new numbers appear immediately.

Expect the question "why 18.66 and not 21.9?" The answer: this is a single
split; 21.9 ± 2.8 is the mean over five. The single split sits inside one
standard deviation. Say it before they ask.

### The more impressive one

```bash
cd IDS_project
..\.venv\Scripts\python.exe -m stress_test.run_packet_roundtrip
```

**2 minutes 34 seconds.** Reads 400 real attack flows out of the 10.5 GB
capture, perturbs the packets, re-extracts features, re-tests. Ends with:

```
THE THREE RUNGS
  unconstrained feature space : 1.0000
  constrained feature space   : 1.0000
  realized in packets         : 0.0000
```

Only start this if you are sure you have the time. Two and a half minutes of
silence in front of a panel is longer than it sounds.

---

## Pre-flight, the morning of

```bash
cd IDS_project
..\.venv\Scripts\python.exe -c "import streamlit, sklearn, xgboost, altair, yaml; print('deps ok')"
..\.venv\Scripts\python.exe -m stress_test.run_constrained_comparison
```

If both pass, the demo works. Then:

- [ ] Close anything else on **port 8501** (old Streamlit sessions survive terminal closes)
- [ ] Open `demo.bat` once and leave it running — first load is the slow one
- [ ] Have the browser already on <http://localhost:8501>
- [ ] Disable sleep/screensaver

---

## What runs, and what does not

| Command | Time | Needs |
|---|---|---|
| `demo.bat` → dashboard | ~20 s | nothing — reports are committed |
| `run_constrained_comparison` | **45 s** | `KDDTrain+.txt` (tracked in git) |
| `run_packet_roundtrip` | **2 m 34 s** | the PCAP + `attack_flows.pkl` (**not** in git) |
| `run_full_matrix` | minutes | `KDDTrain+.txt` |
| `run_multiseed` | minutes — 5 seeds | `KDDTrain+.txt` |
| `run_tree_variance` | minutes — 5 seeds | `KDDTrain+.txt` |
| `run_cicids_audit` | minutes — 200k flows × 3 | CICIDS CSVs (**not** in git) |

Only the first three are demo-safe. The rest are reproduction runs — show
their committed output in the dashboard instead.

---

## Three things that will break it

**1. Running from the wrong folder.** The NSL-KDD loader defaults to
`KDDTrain+.txt` relative to the working directory. Every `python -m
stress_test.*` command must be run from inside `IDS_project/`, not the repo
root.

**2. Running a heavy job while the dashboard is open.** They compete for CPU
and the dashboard stalls on a loading skeleton — which looks like a crash.
Let the job finish, then refresh.

**3. Demoing CICIDS on another machine.** The CSVs and the 10.5 GB PCAP are
not tracked. On a fresh clone, `run_cicids_audit` and `run_packet_roundtrip`
both fail — but the dashboard still shows their committed results, because
the JSON reports are in git. Demo from this machine, or stick to the
dashboard.

---

## If it breaks anyway

| Symptom | Fix |
|---|---|
| Port 8501 in use | `streamlit run dashboard.py --server.port 8502` |
| `ModuleNotFoundError` | The venv is not active. Use the full path: `..\.venv\Scripts\python.exe` |
| `FileNotFoundError: KDDTrain+.txt` | You are in the repo root. `cd IDS_project` |
| Dashboard shows "No report files found" | Not a folder problem — it resolves reports next to `dashboard.py`. The JSONs are genuinely missing: `git checkout IDS_project/*.json` |
| Dashboard numbers look stale | Hard refresh (Ctrl+F5). The cache keys on file mtime but the browser may hold the old page |

**Fallback if everything fails:** the slide deck carries every number, and
`RESULTS_SUMMARY.md` carries them with their source files. Nothing in the
demo is information the deck does not already contain.
