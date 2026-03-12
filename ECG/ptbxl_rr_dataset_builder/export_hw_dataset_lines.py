import argparse
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _torch_load_ckpt(path: Path) -> dict:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _clip_rr(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    if arr.size == 0:
        return arr.astype(np.float32)
    return np.clip(arr, low, high).astype(np.float32)


def _zscore(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    arr = arr.astype(np.float32)
    safe_std = std if std > 0 else 1.0
    return (arr - float(mean)) / float(safe_std)


def _pad_or_trim(arr: np.ndarray, target_len: int, pad_value: float) -> np.ndarray:
    if arr.shape[0] == target_len:
        return arr.astype(np.float32, copy=False)
    if arr.shape[0] > target_len:
        return arr[:target_len].astype(np.float32, copy=False)
    if arr.shape[0] == 0:
        return np.full((target_len,), float(pad_value), dtype=np.float32)
    pad_width = target_len - arr.shape[0]
    return np.pad(arr, (0, pad_width), mode="constant", constant_values=float(pad_value)).astype(np.float32)


def _normalize_hex_line(line: str) -> str:
    return "".join(line.split())


def _decode_f16_words_from_line(line: str, n_words: int) -> np.ndarray:
    s = _normalize_hex_line(line)
    if len(s) != n_words * 4:
        raise RuntimeError(f"bad line length: got {len(s)}, expected {n_words * 4}")
    bits = np.empty((n_words,), dtype=np.uint16)
    for i in range(n_words):
        word = s[i * 4 : (i + 1) * 4]
        bits[i] = struct.unpack("<H", bytes.fromhex(word))[0]
    return bits.view(np.float16)


def _write_hex_words_line(words: list[str], add_spaces: bool) -> str:
    if add_spaces:
        return " ".join(words)
    return "".join(words)


def _rewrite_existing_lines_with_spaces(path: Path, words_per_line: int) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(path, "r", encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
        for line_idx, line in enumerate(fin):
            s = _normalize_hex_line(line)
            expected_len = words_per_line * 4
            if len(s) != expected_len:
                raise RuntimeError(
                    f"bad line length in {path} at line={line_idx}: got {len(s)}, expected {expected_len}"
                )
            words = [s[i * 4 : (i + 1) * 4] for i in range(words_per_line)]
            fout.write(" ".join(words))
            fout.write("\n")
    tmp_path.replace(path)


def _repack_x_lines(hw_dir: Path, add_spaces: bool) -> tuple[Path, Path]:
    manifest_path = hw_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    num_samples = int(manifest["num_samples"])
    seq_len = int(manifest["seq_len"])

    dst0 = hw_dir / "x_ch0_f16_lines.hex"
    dst1 = hw_dir / "x_ch1_f16_lines.hex"

    x_f16_path = None
    files = manifest.get("files", {})
    if isinstance(files, dict) and "x_f16_hex" in files and "path" in files["x_f16_hex"]:
        x_f16_path = hw_dir / str(files["x_f16_hex"]["path"])

    if x_f16_path is None or not x_f16_path.exists():
        if dst0.exists() and dst1.exists():
            if add_spaces:
                _rewrite_existing_lines_with_spaces(dst0, seq_len)
                _rewrite_existing_lines_with_spaces(dst1, seq_len)
            return dst0, dst1
        raise RuntimeError("missing x_f16.hex and x_ch0/x_ch1 also not found; cannot repack x")

    words_per_sample = 2 * seq_len
    with open(x_f16_path, "r", encoding="utf-8") as fin, open(dst0, "w", encoding="utf-8") as f0, open(
        dst1, "w", encoding="utf-8"
    ) as f1:
        for _ in range(num_samples):
            buf = []
            for _ in range(words_per_sample):
                line = fin.readline()
                if not line:
                    raise RuntimeError(f"unexpected EOF while reading {x_f16_path}")
                buf.append(line.strip())
            f0.write(_write_hex_words_line(buf[:seq_len], add_spaces))
            f0.write("\n")
            f1.write(_write_hex_words_line(buf[seq_len:], add_spaces))
            f1.write("\n")

        extra = fin.readline()
        if extra:
            raise RuntimeError(f"extra data after expected {num_samples} samples in {x_f16_path}")

    return dst0, dst1


def _repack_hrv_lines(hw_dir: Path, add_spaces: bool) -> Path:
    manifest_path = hw_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    num_samples = int(manifest["num_samples"])
    hrv_src = hw_dir / manifest["files"]["hrv_f16_hex"]["path"]
    dst = hw_dir / "hrv_f16_lines.hex"

    hrv_shape = manifest["files"]["hrv_f16_hex"].get("shape", [num_samples, 4])
    hrv_dim = int(hrv_shape[1]) if len(hrv_shape) >= 2 else 4

    if not hrv_src.exists():
        if dst.exists():
            if add_spaces:
                _rewrite_existing_lines_with_spaces(dst, hrv_dim)
            return dst
        raise RuntimeError(f"missing {hrv_src}; cannot generate {dst.name}")

    with open(hrv_src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for _ in range(num_samples):
            buf = []
            for _ in range(hrv_dim):
                line = fin.readline()
                if not line:
                    raise RuntimeError(f"unexpected EOF while reading {hrv_src}")
                buf.append(line.strip())
            fout.write(_write_hex_words_line(buf, add_spaces))
            fout.write("\n")

        extra = fin.readline()
        if extra:
            raise RuntimeError(f"extra data after expected {num_samples} samples in {hrv_src}")

    return dst


def _verify_hrv_lines(hw_dir: Path, num_samples_to_check: int, seed: int) -> None:
    manifest_path = hw_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    num_samples = int(manifest["num_samples"])
    hrv_src = hw_dir / manifest["files"]["hrv_f16_hex"]["path"]
    hrv_shape = manifest["files"]["hrv_f16_hex"].get("shape", [num_samples, 4])
    hrv_dim = int(hrv_shape[1]) if len(hrv_shape) >= 2 else 4
    hrv_lines_path = hw_dir / "hrv_f16_lines.hex"

    with open(hrv_lines_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    if len(lines) != num_samples:
        raise RuntimeError(f"line count mismatch: hrv_lines={len(lines)} expected={num_samples}")
    expected_line_len = hrv_dim * 4
    for i, line in enumerate(lines[: min(10, num_samples)]):
        s = _normalize_hex_line(line)
        if len(s) != expected_line_len:
            raise RuntimeError(
                f"bad hrv line length at sample idx={i}: got {len(s)}, expected {expected_line_len}"
            )

    if not hrv_src.exists():
        return

    with open(hrv_src, "r", encoding="utf-8") as f:
        src_words = [line.strip() for line in f]
    if len(src_words) != num_samples * hrv_dim:
        raise RuntimeError(f"word count mismatch: hrv_words={len(src_words)} expected={num_samples * hrv_dim}")

    rng = np.random.RandomState(seed)
    k = min(int(num_samples_to_check), num_samples)
    indices = rng.choice(num_samples, size=k, replace=False).tolist()
    for idx in indices:
        expected = "".join(src_words[int(idx) * hrv_dim : int(idx) * hrv_dim + hrv_dim])
        if _normalize_hex_line(lines[int(idx)]) != expected:
            raise RuntimeError(f"hrv line mismatch at sample idx={idx}")


def _verify_x_lines(hw_dir: Path, num_samples_to_check: int, seed: int) -> None:
    manifest_path = hw_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    seq_len = int(manifest["seq_len"])
    num_samples = int(manifest["num_samples"])
    rr_key = str(manifest.get("rr_key", "rr_ms"))
    idx_csv = Path(str(manifest["index_csv"]))
    ckpt_path = Path(str(manifest["checkpoint_used_for_norm"]))
    rr_clip_low = float(manifest["rr_clip_low"])
    rr_clip_high = float(manifest["rr_clip_high"])

    ch0_path = hw_dir / "x_ch0_f16_lines.hex"
    ch1_path = hw_dir / "x_ch1_f16_lines.hex"

    with open(ch0_path, "r", encoding="utf-8") as f:
        ch0_lines = [line.rstrip("\n") for line in f]
    with open(ch1_path, "r", encoding="utf-8") as f:
        ch1_lines = [line.rstrip("\n") for line in f]

    if len(ch0_lines) != num_samples or len(ch1_lines) != num_samples:
        raise RuntimeError(
            f"line count mismatch: ch0={len(ch0_lines)} ch1={len(ch1_lines)} expected={num_samples}"
        )

    df = pd.read_csv(idx_csv)
    if len(df) != num_samples:
        raise RuntimeError(f"index.csv rows={len(df)} mismatch manifest num_samples={num_samples}")

    ckpt = _torch_load_ckpt(ckpt_path)
    rr_mean = float(ckpt["rr_mean"])
    rr_std = float(ckpt["rr_std"])
    delta_mean = float(ckpt["delta_mean"])
    delta_std = float(ckpt["delta_std"])

    rng = np.random.RandomState(seed)
    k = min(int(num_samples_to_check), num_samples)
    indices = rng.choice(num_samples, size=k, replace=False).tolist()

    for idx in indices:
        file_path = str(df.iloc[int(idx)]["file_path"])
        with np.load(file_path, allow_pickle=True) as data:
            if rr_key in data:
                rr_ms = data[rr_key]
            elif "rr_ms" in data:
                rr_ms = data["rr_ms"]
            else:
                raise RuntimeError(f"no rr array in {file_path}")

        rr_ms = _clip_rr(rr_ms, rr_clip_low, rr_clip_high)
        rr_z = _zscore(rr_ms, rr_mean, rr_std)
        delta_rr = np.diff(rr_ms, prepend=rr_ms[0] if rr_ms.size else 0.0)
        delta_z = _zscore(delta_rr, delta_mean, delta_std)
        rr_z = _pad_or_trim(rr_z, seq_len, pad_value=float(rr_z[-1]) if rr_z.size else 0.0)
        delta_z = _pad_or_trim(delta_z, seq_len, pad_value=0.0)

        expected = np.stack([rr_z, delta_z], axis=0).astype(np.float16, copy=False)
        expected_bits = expected.view(np.uint16)

        ch0 = _decode_f16_words_from_line(ch0_lines[int(idx)], seq_len)
        ch1 = _decode_f16_words_from_line(ch1_lines[int(idx)], seq_len)
        got_bits = np.stack([ch0, ch1], axis=0).view(np.uint16)

        if not np.array_equal(expected_bits, got_bits):
            raise RuntimeError(f"x line mismatch at sample idx={idx}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hw_dir", type=str, default="/root/project/ECG/ptbxl_rr_dataset_builder/output_ptbxl_rhythm_hw_f16")
    parser.add_argument("--skip_x", action="store_true")
    parser.add_argument("--skip_hrv", action="store_true")
    parser.add_argument("--add_spaces", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify_samples", type=int, default=16)
    parser.add_argument("--verify_seed", type=int, default=0)
    args = parser.parse_args()

    hw_dir = Path(args.hw_dir)
    if not args.skip_x:
        x0, x1 = _repack_x_lines(hw_dir, args.add_spaces)
        print("x lines:", x0.name, x1.name)
    if not args.skip_hrv:
        h = _repack_hrv_lines(hw_dir, args.add_spaces)
        print("hrv lines:", h.name)

    if args.verify:
        if not args.skip_x:
            _verify_x_lines(hw_dir, args.verify_samples, args.verify_seed)
            print("verify x: OK")
        if not args.skip_hrv:
            _verify_hrv_lines(hw_dir, args.verify_samples, args.verify_seed)
            print("verify hrv: OK")


if __name__ == "__main__":
    main()
