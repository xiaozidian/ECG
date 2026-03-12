import argparse
import glob
import os
import struct
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

REAL_DATA_ROOT = "/root/project/ECG/ecg_changgen/real_data"
OUT_DIR = "/root/project/ECG/ecg_changgen/train_hospital"
WINDOW_SIZE = 748
WINDOW_PRE = 200
WINDOW_POST = 548


def read_data(file_path):
    num_channels = 3
    fs = 500
    with open(file_path, "rb") as f:
        raw_data = f.read()
    data_int16 = np.frombuffer(raw_data, dtype=np.int16)
    remainder = len(data_int16) % num_channels
    if remainder != 0:
        data_int16 = data_int16[:-remainder]
    data = data_int16.reshape(-1, num_channels)
    return data, fs


def select_best_channel(data):
    best_ch = 0
    max_kurt = -np.inf
    limit = min(len(data), 300000)
    for i in range(data.shape[1]):
        sig = data[:limit, i]
        k = kurtosis(sig)
        if k > max_kurt:
            max_kurt = k
            best_ch = i
    return data[:, best_ch]


def read_ebi_annotations(ebi_path, data_path=None, num_channels=3, fs=500):
    with open(ebi_path, "rb") as f:
        raw = f.read()
    if len(raw) < 32:
        raise ValueError(f"EBI file too small: {ebi_path}")
    magic, record_len, v1, v2, ts = struct.unpack("<5I", raw[:20])
    header_len = 32
    if record_len % 4 != 0:
        raise ValueError(f"Unexpected record_len in EBI: {record_len}")
    if (len(raw) - header_len) % record_len != 0:
        raise ValueError(f"EBI size not aligned to record_len: {ebi_path}")
    n = (len(raw) - header_len) // record_len
    cols = record_len // 4
    arr = np.frombuffer(
        raw,
        dtype=np.uint32,
        offset=header_len,
        count=n * cols,
    ).reshape(n, cols)
    if cols < 6:
        raise ValueError(f"Unexpected EBI record columns: {cols}")
    tick = arr[:, 0].astype(np.uint64)
    aux = arr[:, 1].astype(np.uint32)
    code = arr[:, 2].astype(np.uint32)
    beat_index = arr[:, 3].astype(np.uint32)
    rr_ms = arr[:, 5].astype(np.uint32)
    max_tick = int(tick.max()) if len(tick) else 0
    tick_seconds = 0.005
    if len(tick) > 10:
        dt_tick = np.diff(tick.astype(np.int64))
        rr_s = rr_ms.astype(np.float64) / 1000.0
        rr_s = rr_s[1:]
        m = (dt_tick > 0) & (rr_s > 0) & (rr_s < 5)
        if np.any(m):
            est = rr_s[m] / dt_tick[m].astype(np.float64)
            tick_seconds = float(np.median(est))
    if (
        data_path
        and os.path.exists(data_path)
        and max_tick > 0
        and (tick_seconds <= 0 or not np.isfinite(tick_seconds))
    ):
        data_size = os.path.getsize(data_path)
        total_samples = data_size // (2 * num_channels)
        duration_s = total_samples / float(fs)
        tick_seconds = float(duration_s) / float(max_tick)
    time_s = tick.astype(np.float64) * float(tick_seconds)
    sample_index = np.rint(time_s * float(fs)).astype(np.int64)
    aux_s = aux == np.uint32(131072)
    is_s = aux_s & ((code & np.uint32(536870912)) == 0)
    is_n_from_aux = aux_s & ~is_s
    is_v = (aux == np.uint32(196608)) | ((code & np.uint32(64)) != 0)
    is_n = aux == np.uint32(65536)
    label = np.full(len(code), "Q", dtype=object)
    label[is_n | is_n_from_aux] = "N"
    label[is_v] = "V"
    label[is_s & ~is_v] = "S"
    df = pd.DataFrame(
        {
            "beat_index": beat_index,
            "tick": tick,
            "time_s": time_s,
            "sample_index": sample_index,
            "rr_ms": rr_ms,
            "code": code,
            "aux": aux,
            "label": label,
        }
    )
    meta = {
        "ebi_magic": int(magic),
        "ebi_record_len": int(record_len),
        "ebi_v1": int(v1),
        "ebi_v2": int(v2),
        "ebi_ts": int(ts),
        "tick_seconds": float(tick_seconds),
        "rows": int(len(df)),
    }
    return df, meta


def ensure_beat_annotations(record_dir, data_path, ann_path):
    if os.path.exists(ann_path):
        return True
    dgs_dir = os.path.join(record_dir, "DGS")
    ebi_files = glob.glob(os.path.join(dgs_dir, "*.EBI"))
    if not ebi_files:
        return False
    df, _ = read_ebi_annotations(ebi_files[0], data_path=data_path)
    df["timestamp"] = ""
    df = df[
        [
            "beat_index",
            "tick",
            "time_s",
            "timestamp",
            "sample_index",
            "rr_ms",
            "code",
            "aux",
            "label",
        ]
    ]
    df.to_csv(ann_path, index=False)
    return True


def build_dataset(real_data_root=REAL_DATA_ROOT, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    data_files = glob.glob(
        os.path.join(real_data_root, "*", "data", "*.DATA")
    )
    data_files = sorted(data_files)
    if not data_files:
        raise FileNotFoundError("No .DATA files found.")

    class_map = {"N": 0, "S": 1, "V": 2}
    if WINDOW_PRE + WINDOW_POST != WINDOW_SIZE:
        raise ValueError("WINDOW_PRE + WINDOW_POST must equal WINDOW_SIZE")
    X_list = []
    Y_list = []
    skipped_bounds = 0
    skipped_label = 0
    skipped_missing_ann = 0

    for file_path in data_files:
        record_name = os.path.splitext(os.path.basename(file_path))[0]
        record_dir = os.path.dirname(os.path.dirname(file_path))
        ann_path = os.path.join(
            os.path.dirname(file_path),
            f"{record_name}_beat_annotations.csv",
        )
        if not ensure_beat_annotations(record_dir, file_path, ann_path):
            skipped_missing_ann += 1
            continue

        data, _ = read_data(file_path)
        signal = select_best_channel(data)
        df = pd.read_csv(ann_path)
        if "sample_index" not in df.columns or "label" not in df.columns:
            skipped_missing_ann += 1
            continue

        sample_idx = df["sample_index"].to_numpy()
        labels = df["label"].astype(str).to_numpy()

        for idx, lab in zip(sample_idx, labels):
            if lab not in class_map:
                skipped_label += 1
                continue
            if not np.isfinite(idx):
                skipped_label += 1
                continue
            center = int(idx)
            start = center - WINDOW_PRE
            end = center + WINDOW_POST
            if start < 0 or end > len(signal):
                skipped_bounds += 1
                continue
            segment = signal[start:end].astype(np.float32, copy=False)
            if len(segment) != WINDOW_SIZE:
                skipped_bounds += 1
                continue
            X_list.append(segment)
            Y_list.append(class_map[lab])

    if not X_list:
        raise RuntimeError("No valid beats extracted.")

    X = np.stack(X_list, axis=0).astype(np.float32, copy=False)
    Y = np.asarray(Y_list, dtype=np.int8)
    np.save(os.path.join(out_dir, "X_0.npy"), X)
    np.save(os.path.join(out_dir, "Y_0.npy"), Y)

    return {
        "records": len(data_files) - skipped_missing_ann,
        "samples": int(len(Y)),
        "skipped_bounds": int(skipped_bounds),
        "skipped_label": int(skipped_label),
        "skipped_missing_ann": int(skipped_missing_ann),
        "class_counts": {
            "N": int(np.sum(Y == 0)),
            "S": int(np.sum(Y == 1)),
            "V": int(np.sum(Y == 2)),
        },
    }


def plot_random_segments(
    real_data_root=REAL_DATA_ROOT,
    duration_s=10.0,
    seed=42,
    max_patients=0,
    output_dir=None,
):
    data_files = glob.glob(
        os.path.join(real_data_root, "*", "data", "*.DATA")
    )
    data_files = sorted(data_files)
    if max_patients and max_patients > 0:
        data_files = data_files[: int(max_patients)]
    if not data_files:
        raise FileNotFoundError("No .DATA files found.")
    rng = np.random.default_rng(seed if seed is not None and seed >= 0 else None)
    duration_tag = str(duration_s).replace(".", "p")
    plotted = 0
    skipped = 0
    for file_path in data_files:
        record_name = os.path.splitext(os.path.basename(file_path))[0]
        data, fs = read_data(file_path)
        signal = select_best_channel(data)
        win = int(round(duration_s * fs))
        if win <= 0 or len(signal) <= win:
            skipped += 1
            continue
        start = int(rng.integers(0, len(signal) - win))
        seg = signal[start : start + win]
        t = np.arange(win, dtype=np.float32) / float(fs)
        fig = plt.figure(figsize=(12, 3))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(t, seg, linewidth=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{record_name} random {duration_s}s")
        fig.tight_layout()
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(
                output_dir, f"{record_name}_random_{duration_tag}s.png"
            )
        else:
            out_path = os.path.join(
                os.path.dirname(file_path),
                f"{record_name}_random_{duration_tag}s.png",
            )
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        plotted += 1
    return {
        "patients": len(data_files),
        "plotted": int(plotted),
        "skipped": int(skipped),
    }


def plot_random_segments_with_labels(
    record_dir,
    num_segments=10,
    duration_s=10.0,
    seed=42,
    output_dir=None,
):
    data_dir = os.path.join(record_dir, "data")
    data_files = glob.glob(os.path.join(data_dir, "*.DATA"))
    if not data_files:
        raise FileNotFoundError("No .DATA files found in record dir.")
    data_path = data_files[0]
    record_name = os.path.splitext(os.path.basename(data_path))[0]
    ann_path = os.path.join(data_dir, f"{record_name}_beat_annotations.csv")
    if not os.path.exists(ann_path):
        raise FileNotFoundError("No beat annotations csv found.")
    df = pd.read_csv(ann_path)
    if "sample_index" not in df.columns or "label" not in df.columns:
        raise ValueError("Annotations missing sample_index or label.")
    ann_samples = df["sample_index"].to_numpy(dtype=np.int64)
    ann_labels = df["label"].astype(str).to_numpy()
    data, fs = read_data(data_path)
    signal = select_best_channel(data)
    win = int(round(duration_s * fs))
    if win <= 0 or len(signal) <= win:
        raise ValueError("Invalid duration or too-short signal.")
    rng = np.random.default_rng(seed if seed is not None and seed >= 0 else None)
    starts = rng.integers(0, len(signal) - win, size=int(num_segments))
    duration_tag = str(duration_s).replace(".", "p")
    plotted = 0
    for i, start in enumerate(starts):
        end = int(start + win)
        seg = signal[start:end]
        t = np.arange(win, dtype=np.float32) / float(fs)
        in_win = (ann_samples >= start) & (ann_samples < end)
        win_samples = ann_samples[in_win] - start
        win_labels = ann_labels[in_win]
        fig = plt.figure(figsize=(12, 3))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(t, seg, linewidth=0.8, color="#1f77b4")
        label_colors = {"N": "#2ca02c", "S": "#ff7f0e", "V": "#d62728"}
        label_markers = {"N": "o", "S": "s", "V": "x"}
        shown = set()
        for pos, lab in zip(win_samples, win_labels):
            if lab not in label_colors:
                continue
            x = float(pos) / float(fs)
            y = float(seg[int(pos)])
            ax.scatter(
                x,
                y,
                s=18,
                c=label_colors[lab],
                marker=label_markers[lab],
                label=lab if lab not in shown else None,
                linewidths=0.6,
            )
            ax.text(
                x,
                y,
                lab,
                fontsize=6,
                color=label_colors[lab],
                verticalalignment="bottom",
                horizontalalignment="center",
            )
            shown.add(lab)
        if shown:
            ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{record_name} random {duration_s}s #{i + 1}")
        fig.tight_layout()
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(
                output_dir,
                f"{record_name}_random_{duration_tag}s_{i + 1}.png",
            )
        else:
            out_path = os.path.join(
                data_dir,
                f"{record_name}_random_{duration_tag}s_{i + 1}.png",
            )
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        plotted += 1
    return {
        "record": record_name,
        "segments": int(num_segments),
        "plotted": int(plotted),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-random-10s", action="store_true")
    parser.add_argument("--plot-random-10s-labeled", action="store_true")
    parser.add_argument("--real-data-root", default=REAL_DATA_ROOT)
    parser.add_argument("--record-dir", default=None)
    parser.add_argument("--num-segments", type=int, default=10)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.plot_random_10s_labeled:
        if not args.record_dir:
            raise ValueError("--record-dir is required for labeled plotting.")
        stats = plot_random_segments_with_labels(
            record_dir=args.record_dir,
            num_segments=args.num_segments,
            duration_s=args.duration,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        print(
            "random_plot_labeled="
            f"record:{stats['record']},"
            f"segments:{stats['segments']},"
            f"plotted:{stats['plotted']}"
        )
    elif args.plot_random_10s:
        stats = plot_random_segments(
            real_data_root=args.real_data_root,
            duration_s=args.duration,
            seed=args.seed,
            max_patients=args.max_patients,
            output_dir=args.output_dir,
        )
        print(
            "random_plot="
            f"patients:{stats['patients']},"
            f"plotted:{stats['plotted']},"
            f"skipped:{stats['skipped']}"
        )
    else:
        stats = build_dataset(
            real_data_root=args.real_data_root,
            out_dir=OUT_DIR,
        )
        print(f"records={stats['records']}")
        print(f"samples={stats['samples']}")
        print(
            "class_counts="
            f"N:{stats['class_counts']['N']},"
            f"S:{stats['class_counts']['S']},"
            f"V:{stats['class_counts']['V']}"
        )
        print(
            "skipped="
            f"bounds:{stats['skipped_bounds']},"
            f"label:{stats['skipped_label']},"
            f"missing_ann:{stats['skipped_missing_ann']}"
        )
