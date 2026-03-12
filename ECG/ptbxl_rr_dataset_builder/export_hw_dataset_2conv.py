import argparse
import json
import struct
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

# Constants matching train_rr_cnn_binary_lite_lite.py
CLASSES_SRC = ["SR", "AFIB", "STACH", "SBRAD", "SARRH", "OTHER_RHYTHM"]
CLASSES_BIN = ["SR", "NON_SR"]
BIN_MAP = {"SR": 0, "NON_SR": 1}

def _clip_rr(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    if arr.size == 0:
        return arr.astype(np.float32)
    return np.clip(arr, low, high).astype(np.float32)

def _pad_or_trim(arr: np.ndarray, target_len: int, pad_value: float = None) -> np.ndarray:
    if arr.shape[0] == target_len:
        return arr
    if arr.shape[0] > target_len:
        return arr[:target_len]
    if arr.shape[0] == 0:
        fill_value = 0.0 if pad_value is None else float(pad_value)
        return np.full((target_len,), fill_value, dtype=np.float32)
    pv = float(arr[-1]) if pad_value is None else float(pad_value)
    pad_width = target_len - arr.shape[0]
    return np.pad(arr, (0, pad_width), mode="constant", constant_values=pv).astype(np.float32)

def _load_rr(path: str, rr_key: str, clip_low: float, clip_high: float, seq_len: int) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        if rr_key in data:
            rr_ms = data[rr_key]
        elif "rr_ms" in data:
            rr_ms = data["rr_ms"]
        else:
            raise RuntimeError(f"no rr array in {path}")
    rr_ms = _clip_rr(rr_ms, clip_low, clip_high)
    # Important: Do NOT normalize (z-score) here, just pad/trim raw ms values
    rr_ms = _pad_or_trim(rr_ms, seq_len, pad_value=float(rr_ms[-1]) if rr_ms.size else 0.0)
    return rr_ms.astype(np.float32, copy=False)

def _write_hex_line_f16(arr: np.ndarray, f) -> None:
    # arr: [seq_len], float32
    # Convert to float16, then view as uint16, then write hex
    arr_f16 = arr.astype(np.float16)
    arr_u16 = arr_f16.view(np.uint16)
    
    # Write as continuous hex string per line
    # Each word is 4 chars (16 bits)
    # Total chars per line = seq_len * 4
    hex_str = ""
    for v in arr_u16:
        # Little-endian byte order is standard for these files
        hex_str += struct.pack("<H", int(v)).hex()
    f.write(hex_str + "\n")

def _write_hex_line_u16(val: int, f) -> None:
    # Write a single uint16 value as 4 hex chars
    hex_str = struct.pack("<H", int(val)).hex()
    f.write(hex_str + "\n")

def export_hw_dataset_2conv(
    index_csv: Path,
    meta_csv: Path,
    out_dir: Path,
    seq_len: int,
    rr_key: str,
    rr_clip_low: float,
    rr_clip_high: float
):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset index
    print(f"Loading index from {index_csv}...")
    df = pd.read_csv(index_csv)
    meta = pd.read_csv(meta_csv, usecols=["ecg_id", "patient_id"])
    meta["ecg_id"] = meta["ecg_id"].astype(str)
    df["record_id"] = df["record_id"].astype(str)
    df = df.merge(meta, left_on="record_id", right_on="ecg_id", how="left")
    
    # Filter valid samples (same as training)
    df = df[df["labels"].isin(CLASSES_SRC)].reset_index(drop=True)
    df["bin_label"] = np.where(df["labels"].astype(str) == "SR", "SR", "NON_SR")
    
    num_samples = len(df)
    print(f"Processing {num_samples} samples...")
    
    # Output file paths
    x_path = out_dir / "x_rr_ms_f16.hex"
    y_path = out_dir / "y_u16.hex"
    id_path = out_dir / "record_id.txt"
    manifest_path = out_dir / "manifest.json"
    
    with open(x_path, "w", encoding="utf-8") as f_x, \
         open(y_path, "w", encoding="utf-8") as f_y, \
         open(id_path, "w", encoding="utf-8") as f_id:
        
        for idx in range(num_samples):
            if idx % 1000 == 0:
                print(f"  Processed {idx}/{num_samples}")
                
            row = df.iloc[idx]
            
            # Load and process RR (raw ms)
            rr = _load_rr(row["file_path"], rr_key, rr_clip_low, rr_clip_high, seq_len)
            
            # Get label index
            label_str = str(row["bin_label"])
            label_idx = BIN_MAP[label_str]
            
            # Write files
            _write_hex_line_f16(rr, f_x)
            _write_hex_line_u16(label_idx, f_y)
            f_id.write(f"{row['record_id']}\n")
            
    # Write manifest
    manifest = {
        "num_samples": num_samples,
        "seq_len": seq_len,
        "rr_clip_low": rr_clip_low,
        "rr_clip_high": rr_clip_high,
        "classes": CLASSES_BIN,
        "files": {
            "x": {
                "file": x_path.name,
                "shape": [num_samples, seq_len],
                "dtype": "float16",
                "content": "raw rr intervals in ms (clipped, padded)",
                "note": "one sample per line, little-endian hex"
            },
            "y": {
                "file": y_path.name,
                "shape": [num_samples],
                "dtype": "uint16",
                "content": "binary label index",
                "note": "one sample per line, little-endian hex"
            },
            "id": {
                "file": id_path.name,
                "content": "record_id string"
            }
        }
    }
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Done. Output saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_csv", type=str, default="/root/project/ECG/ptbxl_rr_dataset_builder/output_ptbxl_rhythm/index.csv")
    parser.add_argument("--meta_csv", type=str, default="/root/project/ECG/PTB-XL/ptbxl_database.csv")
    parser.add_argument("--out_dir", type=str, default="/root/project/ECG/ptbxl_rr_dataset_builder/output_ptbxl_rhythm_hw_f16_2conv")
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--rr_key", type=str, default="rr_ms")
    parser.add_argument("--rr_clip_low", type=float, default=300.0)
    parser.add_argument("--rr_clip_high", type=float, default=2000.0)
    
    args = parser.parse_args()
    
    export_hw_dataset_2conv(
        Path(args.index_csv),
        Path(args.meta_csv),
        Path(args.out_dir),
        args.seq_len,
        args.rr_key,
        args.rr_clip_low,
        args.rr_clip_high
    )
