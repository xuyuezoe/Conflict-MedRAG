#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def read_labels(path: Path, sample_col: str, label_col: str):
    m = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sid = r[sample_col].strip()
            label = r[label_col].strip()
            if sid and label:
                m[sid] = label
    return m


def cohens_kappa(a, b):
    keys = sorted(set(a.keys()) & set(b.keys()))
    if not keys:
        raise ValueError("没有重叠样本")

    labels = sorted(set(a[k] for k in keys) | set(b[k] for k in keys))
    n = len(keys)
    po = sum(1 for k in keys if a[k] == b[k]) / n

    pa = {lab: sum(1 for k in keys if a[k] == lab) / n for lab in labels}
    pb = {lab: sum(1 for k in keys if b[k] == lab) / n for lab in labels}
    pe = sum(pa[lab] * pb[lab] for lab in labels)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ann1", required=True)
    p.add_argument("--ann2", required=True)
    p.add_argument("--sample-col", default="sample_id")
    p.add_argument("--label-col", default="sc_fc_label")
    args = p.parse_args()

    a = read_labels(Path(args.ann1), args.sample_col, args.label_col)
    b = read_labels(Path(args.ann2), args.sample_col, args.label_col)

    k = cohens_kappa(a, b)
    print(f"Cohen's kappa = {k:.4f}")


if __name__ == "__main__":
    main()
