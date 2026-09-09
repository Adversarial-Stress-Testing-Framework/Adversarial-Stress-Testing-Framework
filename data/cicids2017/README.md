# CICIDS2017

Not tracked in git - roughly 1 GB of CSVs and 8-10 GB per PCAP.

Download from https://www.unb.ca/cic/datasets/ids-2017.html

## For the dataset audit (Phase 2)

`MachineLearningCSV.zip` (~230 MB). Extract the eight day CSVs into this
directory:

    data/cicids2017/Monday-WorkingHours.pcap_ISCX.csv
    data/cicids2017/Tuesday-WorkingHours.pcap_ISCX.csv
    ...

Then:

    cd IDS_project
    python -m stress_test.run_cicids_audit

## For packet-level realizability (Phase 3)

`Tuesday-WorkingHours.pcap` (~8-10 GB). Tuesday rather than Wednesday: it is
FTP/SSH brute force, so the flows are short and well defined, and padding and
delay are realistic evasions against them. Wednesday's DoS capture is larger
and its flows are harder to reason about.

Phase 3 also needs a JDK for CICFlowMeter - Temurin 11 or 17 from adoptium.net.
