"""
Pull the brute-force attack flows out of Tuesday's capture.

Phase 3 needs real attack traffic to perturb, so this streams the 10.5 GB
capture once and keeps only packets belonging to the FTP-Patator and
SSH-Patator flows: 172.16.0.1 attacking 192.168.10.50 on ports 21 and 22.
Everything else is discarded as it goes, because holding the file in memory is
not an option.

Flows are keyed by the usual 5-tuple with direction normalised, so a flow and
its reverse land together. A flow is considered finished when it has seen a FIN
or RST in either direction; brute-force attempts are short, so most complete
quickly.

Run with: python -m stress_test.pcap_extract
"""

import json
import pickle
import time
from collections import defaultdict

ATTACKER = "172.16.0.1"
VICTIM = "192.168.10.50"
PORTS = (21, 22)
PCAP = "../data/cicids2017/Tuesday-WorkingHours.pcap"
OUT = "../data/cicids2017/attack_flows.pkl"
MAX_FLOWS = 400
MAX_PACKETS_PER_FLOW = 400


def flow_key(ip, tcp):
    """Direction-normalised 5-tuple, so both halves of a conversation match."""
    a = (ip.src, tcp.sport)
    b = (ip.dst, tcp.dport)
    return (a, b) if a <= b else (b, a)


def main():
    from scapy.all import PcapReader, TCP, IP

    flows = defaultdict(list)
    done = set()
    seen = 0
    kept = 0
    t0 = time.time()

    with PcapReader(PCAP) as reader:
        for pkt in reader:
            seen += 1
            if seen % 1_000_000 == 0:
                print(f"  {seen:,} packets, {len(flows)} flows, "
                      f"{len(done)} complete, {time.time() - t0:.0f}s", flush=True)

            if IP not in pkt or TCP not in pkt:
                continue
            ip, tcp = pkt[IP], pkt[TCP]

            # Either direction of the attacker/victim conversation on a
            # brute-forced service.
            fwd = ip.src == ATTACKER and ip.dst == VICTIM and tcp.dport in PORTS
            rev = ip.src == VICTIM and ip.dst == ATTACKER and tcp.sport in PORTS
            if not (fwd or rev):
                continue

            k = flow_key(ip, tcp)
            if k in done:
                continue

            # Store only what the feature extractor needs. Keeping scapy packet
            # objects for hundreds of flows exhausts memory on a capture this
            # size, and the payload length matters, not the payload.
            flows[k].append({
                "t": float(pkt.time),
                "fwd": bool(fwd),
                "len": len(pkt),
                "payload": len(tcp.payload),
                "flags": str(tcp.flags),
                "hdr": tcp.dataofs * 4,
                "win": int(tcp.window),
                "dport": int(tcp.dport if fwd else tcp.sport),
            })
            kept += 1

            f = str(tcp.flags)
            if "F" in f or "R" in f or len(flows[k]) >= MAX_PACKETS_PER_FLOW:
                done.add(k)

            if len(done) >= MAX_FLOWS:
                print(f"  reached {MAX_FLOWS} complete flows, stopping early")
                break

    complete = {k: v for k, v in flows.items() if k in done and len(v) >= 4}
    print(f"\nscanned {seen:,} packets in {time.time() - t0:.0f}s")
    print(f"kept {kept:,} packets across {len(flows)} flows")
    print(f"{len(complete)} complete flows with >= 4 packets")

    by_port = defaultdict(int)
    for v in complete.values():
        by_port[v[0]["dport"]] += 1
    print(f"by service: {dict(by_port)}")

    with open(OUT, "wb") as fh:
        pickle.dump({str(k): v for k, v in complete.items()}, fh)
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
