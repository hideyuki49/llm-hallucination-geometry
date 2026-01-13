#!/usr/bin/env python3
# extract_rates.py
import re
import csv
import argparse
from pathlib import Path

RE_HEADER = re.compile(
    r"^====\s+mode=(?P<mode>[^|]+)\s+\|\s+backend=(?P<backend>[^|]+)\s+\|\s+model_id=(?P<model_id>[^|]+)\s+\|\s+regime=(?P<regime>[^|]+)\s+\|\s+repeat_fake=(?P<repeat_fake>\d+)\s+\|\s+NOTE_ORDER=(?P<note_order>[^|]+)\s+\|\s+MISSING_INCLUDE_FAKES=(?P<missing_include_fakes>\d+)\s+====\s*$"
)

RE_ROW = re.compile(
    r"^fake=\s*(?P<fake>\d+)\s*\|\s*"
    r"A=(?P<A_num>\d+)/(?P<A_den>\d+)\s*\|\s*"
    r"F=(?P<F_num>\d+)/(?P<F_den>\d+)\s*\|\s*"
    r"H_any=(?P<H_num>\d+)/(?P<H_den>\d+)\s*\|\s*"
    r"H_missFP=(?P<Hm_num>\d+)/(?P<Hm_den>\d+)\s*\|\s*"
    r"U=(?P<U_num>\d+)/(?P<U_den>\d+)\s*\|"
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_txt", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    cur = None
    rows = []

    for line in Path(args.in_txt).read_text(encoding="utf-8", errors="ignore").splitlines():
        m = RE_HEADER.match(line.strip())
        if m:
            cur = m.groupdict()
            continue

        m = RE_ROW.match(line.strip())
        if m and cur:
            gd = m.groupdict()
            # n: use H_any denominator (should match others; keep as integer)
            n = int(gd["H_den"])
            rows.append({
                **cur,
                "fake": int(gd["fake"]),
                "n": n,
                "H_any": int(gd["H_num"]),
                "A": int(gd["A_num"]),
                "U": int(gd["U_num"]),
                "F": int(gd["F_num"]),
            })

    fieldnames = [
        "backend","model_id","regime","mode","repeat_fake","note_order","missing_include_fakes",
        "fake","n","H_any","A","U","F"
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

if __name__ == "__main__":
    main()
