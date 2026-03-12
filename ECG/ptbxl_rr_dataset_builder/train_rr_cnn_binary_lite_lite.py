import argparse
import json
import struct
from pathlib import Path
from typing import Dict, Tuple
 
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
 
 
CLASSES_SRC = ["SR", "AFIB", "STACH", "SBRAD", "SARRH", "OTHER_RHYTHM"]
CLASSES_BIN = ["SR", "NON_SR"]
BIN_MAP = {"SR": 0, "NON_SR": 1}
 
 
def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 

def _torch_load_ckpt(path: Path) -> dict:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def export_ckpt_weights_hex_f16(ckpt_path: Path, out_dir: Path) -> Path:
    ckpt_path = ckpt_path.resolve()
    out_dir = out_dir.resolve()
    ckpt = _torch_load_ckpt(ckpt_path)
    state = ckpt.get("state_dict", ckpt)
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"bad checkpoint contents: {ckpt_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, dict] = {}
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        file_name = f"{name}.hex"
        out_path = out_dir / file_name

        t_f16 = tensor.detach().cpu().to(torch.float16).contiguous()
        np_f16 = t_f16.numpy()
        u16 = np_f16.view(np.uint16).reshape(-1)

        with open(out_path, "w", encoding="utf-8") as f:
            for v in u16.tolist():
                f.write(struct.pack("<H", int(v)).hex())
                f.write("\n")

        tensors[str(name)] = {
            "file": file_name,
            "shape": list(tensor.shape),
            "dtype": "float16",
            "numel": int(tensor.numel()),
        }

    manifest = {
        "checkpoint": str(ckpt_path),
        "format": {
            "encoding": "ieee754_float16",
            "endianness": "little",
            "word_bytes": 2,
            "hex_chars_per_word": 4,
            "layout": "flatten_c_order",
            "one_word_per_line": True,
            "note": "每行是一个16-bit word的byte序列(小端)的hex字符串",
        },
        "tensors": tensors,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    for name, spec in tensors.items():
        path = out_dir / spec["file"]
        with open(path, "r", encoding="utf-8") as f:
            n_lines = 0
            for line_idx, line in enumerate(f):
                s = line.strip()
                if len(s) != 4:
                    raise RuntimeError(f"bad hex word length: {path} line={line_idx} got={len(s)}")
                n_lines += 1
        if n_lines != int(spec["numel"]):
            raise RuntimeError(f"numel mismatch: {name} file_lines={n_lines} expected={spec['numel']}")

    return out_dir

 
def _load_index(index_csv: Path, meta_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(index_csv)
    meta = pd.read_csv(meta_csv, usecols=["ecg_id", "patient_id"])
    meta["ecg_id"] = meta["ecg_id"].astype(str)
    df["record_id"] = df["record_id"].astype(str)
    df = df.merge(meta, left_on="record_id", right_on="ecg_id", how="left")
    if df["patient_id"].isna().any():
        missing = df[df["patient_id"].isna()]["record_id"].head(5).tolist()
        raise RuntimeError(f"missing patient_id for records: {missing}")
    df = df[df["labels"].isin(CLASSES_SRC)].reset_index(drop=True)
    df["bin_label"] = np.where(df["labels"].astype(str) == "SR", "SR", "NON_SR")
    return df
 
 
def _split_by_patient(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patients = df["patient_id"].unique().tolist()
    rng = np.random.RandomState(seed)
    rng.shuffle(patients)
    n_total = len(patients)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))
    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train : n_train + n_val])
    test_patients = set(patients[n_train + n_val :])
    train_df = df[df["patient_id"].isin(train_patients)].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_patients)].reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_patients)].reset_index(drop=True)
    return train_df, val_df, test_df
 
 
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
 
 
class RRBinaryDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        rr_key: str,
        rr_clip_low: float,
        rr_clip_high: float,
    ) -> None:
        self.df = df
        self.seq_len = int(seq_len)
        self.rr_key = rr_key
        self.rr_clip_low = float(rr_clip_low)
        self.rr_clip_high = float(rr_clip_high)
 
    def __len__(self) -> int:
        return len(self.df)
 
    def _load_rr(self, path: str) -> np.ndarray:
        with np.load(path, allow_pickle=True) as data:
            if self.rr_key in data:
                rr_ms = data[self.rr_key]
            elif "rr_ms" in data:
                rr_ms = data["rr_ms"]
            else:
                raise RuntimeError(f"no rr array in {path}")
        rr_ms = _clip_rr(rr_ms, self.rr_clip_low, self.rr_clip_high)
        rr_ms = _pad_or_trim(rr_ms, self.seq_len, pad_value=float(rr_ms[-1]) if rr_ms.size else 0.0)
        return rr_ms.astype(np.float32, copy=False)
 
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        rr = self._load_rr(row["file_path"])
        label = BIN_MAP[str(row["bin_label"])]
        x = torch.from_numpy(rr[None, :])
        y = torch.tensor(label, dtype=torch.long)
        return x, y
 
 
class RRLite(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, pool_len: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=3, kernel_size=7, stride=1, padding=3, bias=False)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=2, bias=False)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.adapt_pool = nn.AdaptiveAvgPool1d(pool_len)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * pool_len, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.pool1(x)
        x = self.act(self.conv2(x))
        x = self.pool2(x)
        x = self.adapt_pool(x)
        x = self.flatten(x)
        return self.fc(x)
 
 
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = nn.functional.log_softmax(logits, dim=1)
        logpt = logp.gather(1, target.view(-1, 1)).squeeze(1)
        pt = logpt.exp()
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.weight is not None:
            weight = self.weight.gather(0, target)
            loss = loss * weight
        return loss.mean()


def _class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    total = float(counts.sum())
    weights = total / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)
 
 
def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm
 
 
def _metrics_from_cm(cm: np.ndarray) -> Dict[str, np.ndarray]:
    tp = np.diag(cm).astype(np.float32)
    pred_sum = cm.sum(axis=0).astype(np.float32)
    true_sum = cm.sum(axis=1).astype(np.float32)
    precision = tp / np.maximum(pred_sum, 1.0)
    recall = tp / np.maximum(true_sum, 1.0)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    macro_f1 = float(np.mean(f1))
    return {"precision": precision, "recall": recall, "f1": f1, "macro_f1": macro_f1}
 
 
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    all_preds = []
    all_labels = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(pred)
        all_labels.append(y.numpy())
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    cm = _confusion_matrix(y_true, y_pred, len(CLASSES_BIN))
    metrics = _metrics_from_cm(cm)
    metrics["confusion_matrix"] = cm
    return metrics
 
 
def _print_split_counts(name: str, dff: pd.DataFrame) -> None:
    counts = dff["bin_label"].value_counts().reindex(CLASSES_BIN, fill_value=0)
    total = int(counts.sum())
    print(f"\n[{name}] total={total}")
    for k, v in counts.items():
        print(f"  {k:8s}: {int(v)}")
 
 
def train(args: argparse.Namespace) -> None:
    _set_seed(args.seed)
    index_csv = Path(args.index_csv)
    meta_csv = Path(args.meta_csv)
    df = _load_index(index_csv, meta_csv)
    train_df, val_df, test_df = _split_by_patient(df, 0.8, 0.1, args.seed)
    _print_split_counts("train", train_df)
    _print_split_counts("val", val_df)
    _print_split_counts("test", test_df)
 
    train_ds = RRBinaryDataset(
        train_df,
        args.seq_len,
        args.rr_key,
        args.rr_clip_low,
        args.rr_clip_high,
    )
    val_ds = RRBinaryDataset(
        val_df,
        args.seq_len,
        args.rr_key,
        args.rr_clip_low,
        args.rr_clip_high,
    )
    test_ds = RRBinaryDataset(
        test_df,
        args.seq_len,
        args.rr_key,
        args.rr_clip_low,
        args.rr_clip_high,
    )
 
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
 
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
 
    model = RRLite(1, len(CLASSES_BIN), pool_len=args.lite_pool_len).to(device)
 
    y_train = train_df["bin_label"].map(BIN_MAP).to_numpy()
    weights = _class_weights(y_train, len(CLASSES_BIN)).to(device)
    if args.loss == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma, weight=weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
 
    best_macro = -1.0
    best_state = None
 
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * x.size(0)
        avg_loss = total_loss / max(len(train_loader.dataset), 1)
        val_metrics = evaluate(model, val_loader, device)
        macro_f1 = float(val_metrics["macro_f1"])
        if macro_f1 > best_macro:
            best_macro = macro_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"epoch={epoch} loss={avg_loss:.6f} val_macro_f1={macro_f1:.4f}")
 
    if best_state is not None and args.save_path:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_state,
                "classes": CLASSES_BIN,
                "seq_len": args.seq_len,
                "rr_key": args.rr_key,
                "rr_clip_low": args.rr_clip_low,
                "rr_clip_high": args.rr_clip_high,
                "lite_pool_len": args.lite_pool_len,
            },
            args.save_path,
        )
 
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    print("test_macro_f1", float(test_metrics["macro_f1"]))
    print("test_f1_per_class", dict(zip(CLASSES_BIN, test_metrics["f1"].tolist())))
    print("test_confusion_matrix")
    print(test_metrics["confusion_matrix"])
 
 
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_csv", type=str, default="/root/project/ECG/ptbxl_rr_dataset_builder/output_ptbxl_rhythm/index.csv")
    parser.add_argument("--meta_csv", type=str, default="/root/project/ECG/PTB-XL/ptbxl_database.csv")
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--rr_key", type=str, default="rr_ms")
    parser.add_argument("--lite_pool_len", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", type=str, choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--save_path",
        type=str,
        default="/root/project/ECG/ptbxl_rr_dataset_builder/checkpoints/rr_cnn_binary_lite_2conv.pt",
    )
    parser.add_argument("--rr_clip_low", type=float, default=300.0)
    parser.add_argument("--rr_clip_high", type=float, default=2000.0)
    parser.add_argument("--export_ckpt_hex_f16", type=str, default="")
    parser.add_argument("--export_out_dir", type=str, default="")
    return parser
 
 
if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.export_ckpt_hex_f16:
        ckpt_path = Path(args.export_ckpt_hex_f16)
        out_dir = Path(args.export_out_dir) if args.export_out_dir else ckpt_path.parent / f"{ckpt_path.stem}_hex_f16"
        out = export_ckpt_weights_hex_f16(ckpt_path, out_dir)
        print("export hex f16:", str(out))
    else:
        train(args)
