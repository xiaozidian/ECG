import argparse
import ast
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import importlib.util


def _has_wfdb():
    return importlib.util.find_spec("wfdb") is not None


def _load_wfdb_signal(record_path):
    import wfdb
    signal, meta = wfdb.rdsamp(record_path)
    fs = float(meta.get("fs", 500))
    return signal, fs


def _best_lead(signal):
    from scipy.stats import kurtosis
    if signal.ndim == 1:
        return signal, 0
    best_idx = 0
    best_k = -np.inf
    for i in range(signal.shape[1]):
        k = kurtosis(signal[:, i])
        if k > best_k:
            best_k = k
            best_idx = i
    return signal[:, best_idx], int(best_idx)


def _extract_rr(signal, fs):
    import neurokit2 as nk
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)
    r_idx = np.asarray(rpeaks["ECG_R_Peaks"]).astype(int)
    if len(r_idx) < 2:
        return np.array([], dtype=np.float32)
    rr_ms = np.diff(r_idx) / float(fs) * 1000.0
    return rr_ms.astype(np.float32)


def _parse_scp_codes(val):
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return {}
    return {}


def _build_target_code_map():
    return {
        "NORM": ["NORM"],
        "AFIB": ["AFIB"],
        "AFL": ["AFLT"],
        "SARRH": ["SARRH"],
        "SB": ["SBRAD"],
        "ST": ["STACH"],
        "PVC_VF": ["PVC", "VFL", "VF"]
    }


def _label_rhythm(scp_codes):
    keys = set(scp_codes.keys())
    other_codes = {"AFLT", "SVARR", "SVTAC", "PSVT", "PACE", "BIGU", "TRIGU"}
    if "AFIB" in keys:
        return "AFIB"
    if keys & other_codes:
        return "OTHER_RHYTHM"
    if "STACH" in keys:
        return "STACH"
    if "SBRAD" in keys:
        return "SBRAD"
    if "SARRH" in keys:
        return "SARRH"
    if "SR" in keys:
        return "SR"
    return None


def _label_ectopy(scp_codes, include_bigu_trigu):
    keys = set(scp_codes.keys())
    ectopy_codes = {"PVC"}
    if include_bigu_trigu:
        ectopy_codes = ectopy_codes | {"BIGU", "TRIGU"}
    return "ECTOPY" if keys & ectopy_codes else "NO_ECTOPY"


def _match_labels(scp_codes, target_map):
    matched = []
    keys = set(scp_codes.keys())
    for label, codes in target_map.items():
        if any(code in keys for code in codes):
            matched.append(label)
    return matched


def _resolve_record_path(records_dir: Path, rel_path: str) -> Path:
    rel_path = rel_path.strip()
    prefixes = [
        "records500/",
        "ptbxl_records500/",
        "records100/",
        "ptbxl_records100/",
    ]
    for p in prefixes:
        if rel_path.startswith(p):
            rel_path = rel_path[len(p):]
            break
    return records_dir / rel_path


def _build_dataset_from_labeler(records_dir, metadata_csv, output_dir, min_per_class, labeler, class_names=None):
    records_dir = Path(records_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not _has_wfdb():
        raise RuntimeError("wfdb is not installed")
    if not os.path.exists(metadata_csv):
        index_path = output_dir / "index.csv"
        pd.DataFrame([]).to_csv(index_path, index=False)
        summary = {
            "total_records": 0,
            "class_counts": {},
            "min_required_per_class": int(min_per_class),
            "insufficient_classes": [],
            "missing_files": 0,
            "skipped_short_rr": 0,
            "missing_metadata": str(metadata_csv)
        }
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return index_path, summary_path
    df = pd.read_csv(metadata_csv)
    if "scp_codes" not in df.columns:
        raise ValueError("scp_codes column not found in metadata")
    if "filename_hr" in df.columns:
        path_col = "filename_hr"
    elif "filename_lr" in df.columns:
        path_col = "filename_lr"
    else:
        raise ValueError("filename_hr or filename_lr column not found in metadata")
    results = []
    if class_names:
        counts = {k: 0 for k in class_names}
    else:
        counts = {}
    missing_files = []
    skipped_short = []
    for _, row in df.iterrows():
        scp_codes = _parse_scp_codes(row["scp_codes"])
        label = labeler(scp_codes)
        if not label:
            continue
        labels = [label]
        rel_path = str(row[path_col])
        record_path = _resolve_record_path(records_dir, rel_path)
        dat_path = record_path.with_suffix(".dat")
        hea_path = record_path.with_suffix(".hea")
        if not dat_path.exists() or not hea_path.exists():
            missing_files.append(rel_path)
            continue
        if dat_path.stat().st_size == 0 or hea_path.stat().st_size == 0:
            missing_files.append(rel_path)
            continue
        signal, fs = _load_wfdb_signal(str(record_path))
        lead_signal, lead_idx = _best_lead(signal)
        rr_ms = _extract_rr(lead_signal, fs)
        if rr_ms.size < 3:
            skipped_short.append(rel_path)
            continue
        rr_mean = float(rr_ms.mean())
        rr_std = float(rr_ms.std())
        rr_centered = rr_ms - rr_mean
        delta_rr = np.diff(rr_ms)
        if rr_mean > 0:
            rr_cv = rr_std / rr_mean
        else:
            rr_cv = float("nan")
        if delta_rr.size > 0:
            rmssd = float(np.sqrt(np.mean(delta_rr.astype(np.float64) ** 2)))
        else:
            rmssd = float("nan")
        record_id = str(row["ecg_id"]) if "ecg_id" in row else Path(rel_path).stem
        out_path = output_dir / f"{record_id}.npz"
        np.savez(
            out_path,
            rr_ms=rr_ms,
            rr_centered=rr_centered.astype(np.float32),
            delta_rr=delta_rr.astype(np.float32),
            labels=np.array(labels, dtype=object),
            fs=float(fs),
            lead_index=int(lead_idx)
        )
        for lab in labels:
            counts[lab] = counts.get(lab, 0) + 1
        results.append({
            "record_id": record_id,
            "labels": ",".join(labels),
            "rr_len": int(rr_ms.size),
            "rr_mean": rr_mean,
            "rr_std": rr_std,
            "rr_cv": float(rr_cv),
            "rmssd": float(rmssd),
            "delta_len": int(delta_rr.size),
            "file_path": str(out_path)
        })
    df_out = pd.DataFrame(results)
    index_path = output_dir / "index.csv"
    df_out.to_csv(index_path, index=False)
    summary = {
        "total_records": int(len(df_out)),
        "class_counts": counts,
        "min_required_per_class": int(min_per_class),
        "insufficient_classes": [k for k, v in counts.items() if v < min_per_class],
        "missing_files": int(len(missing_files)),
        "skipped_short_rr": int(len(skipped_short))
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return index_path, summary_path


def build_dataset(records_dir, metadata_csv, output_dir, min_per_class):
    target_map = _build_target_code_map()

    def labeler(scp_codes):
        labels = _match_labels(scp_codes, target_map)
        return labels[0] if labels else None

    return _build_dataset_from_labeler(
        records_dir=records_dir,
        metadata_csv=metadata_csv,
        output_dir=output_dir,
        min_per_class=min_per_class,
        labeler=labeler,
        class_names=list(target_map.keys())
    )


def build_rhythm_dataset(records_dir, metadata_csv, output_dir, min_per_class):
    class_names = ["SR", "AFIB", "STACH", "SBRAD", "SARRH", "OTHER_RHYTHM"]
    return _build_dataset_from_labeler(
        records_dir=records_dir,
        metadata_csv=metadata_csv,
        output_dir=output_dir,
        min_per_class=min_per_class,
        labeler=_label_rhythm,
        class_names=class_names
    )


def build_ectopy_dataset(records_dir, metadata_csv, output_dir, min_per_class, include_bigu_trigu):
    class_names = ["NO_ECTOPY", "ECTOPY"]
    return _build_dataset_from_labeler(
        records_dir=records_dir,
        metadata_csv=metadata_csv,
        output_dir=output_dir,
        min_per_class=min_per_class,
        labeler=lambda scp_codes: _label_ectopy(scp_codes, include_bigu_trigu),
        class_names=class_names
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records_dir", type=str, default="/root/project/ECG/PTB-XL/ptbxl_records500")
    parser.add_argument("--metadata_csv", type=str, default="/root/project/ECG/PTB-XL/ptbxl_database.csv")
    parser.add_argument("--output_dir", type=str, default="/root/project/ECG/ptbxl_rr_dataset_builder/output_ptbxl")
    parser.add_argument("--output_dir_rhythm", type=str, default="/root/project/ECG/ptbxl_rr_dataset_builder/output_ptbxl_rhythm")
    parser.add_argument("--output_dir_ectopy", type=str, default="/root/project/ECG/ptbxl_rr_dataset_builder/output_ptbxl_ectopy")
    parser.add_argument("--task", type=str, default="default", choices=["default", "rhythm", "ectopy", "all"])
    parser.add_argument("--ectopy_include_bigu_trigu", action="store_true")
    parser.add_argument("--min_per_class", type=int, default=200)
    args = parser.parse_args()
    if args.task == "default":
        index_path, summary_path = build_dataset(
            records_dir=args.records_dir,
            metadata_csv=args.metadata_csv,
            output_dir=args.output_dir,
            min_per_class=args.min_per_class
        )
        print(f"Index saved to: {index_path}")
        print(f"Summary saved to: {summary_path}")
        return
    if args.task in {"rhythm", "all"}:
        index_path, summary_path = build_rhythm_dataset(
            records_dir=args.records_dir,
            metadata_csv=args.metadata_csv,
            output_dir=args.output_dir_rhythm,
            min_per_class=args.min_per_class
        )
        print(f"Rhythm index saved to: {index_path}")
        print(f"Rhythm summary saved to: {summary_path}")
    if args.task in {"ectopy", "all"}:
        index_path, summary_path = build_ectopy_dataset(
            records_dir=args.records_dir,
            metadata_csv=args.metadata_csv,
            output_dir=args.output_dir_ectopy,
            min_per_class=args.min_per_class,
            include_bigu_trigu=args.ectopy_include_bigu_trigu
        )
        print(f"Ectopy index saved to: {index_path}")
        print(f"Ectopy summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
