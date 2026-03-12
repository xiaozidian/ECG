import argparse
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
 
 
def _zscore(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    safe_std = np.where(std > 0, std, 1.0)
    return (arr - mean) / safe_std
 
 
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
 
 
def _sample_entropy(x: np.ndarray, m: int = 2, r: float = None) -> float:
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    if n <= m + 1:
        return 0.0
    std = float(np.std(x))
    if r is None:
        r = 0.2 * std
    if r <= 0:
        return 0.0
 
    def _count_matches(length: int) -> int:
        count = 0
        for i in range(n - length):
            xi = x[i : i + length]
            for j in range(i + 1, n - length + 1):
                xj = x[j : j + length]
                if np.max(np.abs(xi - xj)) < r:
                    count += 1
        return count
 
    b = _count_matches(m)
    a = _count_matches(m + 1)
    if b == 0 or a == 0:
        return 0.0
    return float(-np.log(a / b))
 
 
def _hrv_features(rr_ms: np.ndarray) -> np.ndarray:
    rr_ms = np.asarray(rr_ms, dtype=np.float32)
    if rr_ms.size == 0:
        return np.zeros((4,), dtype=np.float32)
    sdnn = float(np.std(rr_ms))
    diff = np.diff(rr_ms)
    if diff.size == 0:
        rmssd = 0.0
        pnn50 = 0.0
    else:
        rmssd = float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)))
        pnn50 = float(np.mean(np.abs(diff) > 50.0))
    sampen = _sample_entropy(rr_ms, m=2, r=0.2 * float(np.std(rr_ms)) if rr_ms.size > 1 else 0.0)
    return np.array([sdnn, rmssd, pnn50, sampen], dtype=np.float32)
 
 
class RRBinaryDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        rr_key: str,
        rr_mean: float,
        rr_std: float,
        delta_mean: float,
        delta_std: float,
        hrv_mean: np.ndarray,
        hrv_std: np.ndarray,
        rr_clip_low: float,
        rr_clip_high: float,
    ) -> None:
        self.df = df
        self.seq_len = int(seq_len)
        self.rr_key = rr_key
        self.rr_mean = float(rr_mean)
        self.rr_std = float(rr_std)
        self.delta_mean = float(delta_mean)
        self.delta_std = float(delta_std)
        self.hrv_mean = np.asarray(hrv_mean, dtype=np.float32)
        self.hrv_std = np.asarray(hrv_std, dtype=np.float32)
        self.rr_clip_low = float(rr_clip_low)
        self.rr_clip_high = float(rr_clip_high)
 
    def __len__(self) -> int:
        return len(self.df)
 
    def _load_rr(self, path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.load(path, allow_pickle=True) as data:
            if self.rr_key in data:
                rr_ms = data[self.rr_key]
            elif "rr_ms" in data:
                rr_ms = data["rr_ms"]
            else:
                raise RuntimeError(f"no rr array in {path}")
        rr_ms = _clip_rr(rr_ms, self.rr_clip_low, self.rr_clip_high)
        hrv = _hrv_features(rr_ms)
        hrv = _zscore(hrv, self.hrv_mean, self.hrv_std)
        rr_z = _zscore(rr_ms, self.rr_mean, self.rr_std)
        delta_rr = np.diff(rr_ms, prepend=rr_ms[0] if rr_ms.size else 0.0)
        delta_z = _zscore(delta_rr, self.delta_mean, self.delta_std)
        rr_z = _pad_or_trim(rr_z, self.seq_len, pad_value=rr_z[-1] if rr_z.size else 0.0)
        delta_z = _pad_or_trim(delta_z, self.seq_len, pad_value=0.0)
        return rr_z, delta_z, hrv
 
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        rr, delta, hrv = self._load_rr(row["file_path"])
        label = BIN_MAP[str(row["bin_label"])]
        x = torch.from_numpy(np.stack([rr, delta], axis=0))
        hrv = torch.from_numpy(hrv)
        y = torch.tensor(label, dtype=torch.long)
        return x, hrv, y
 
 
class RRLite(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hrv_dim: int, pool_len: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=3, kernel_size=7, stride=1, padding=3, bias=False)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=2, bias=False)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=12, kernel_size=5, stride=1, padding=2, bias=False)
        self.pool3 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(in_channels=12, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.adapt_pool = nn.AdaptiveAvgPool1d(pool_len)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * pool_len + hrv_dim, 64)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, num_classes)
 
    def forward(self, x: torch.Tensor, hrv: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.pool1(x)
        x = self.act(self.conv2(x))
        x = self.pool2(x)
        x = self.act(self.conv3(x))
        x = self.pool3(x)
        x = self.act(self.conv4(x))
        x = self.adapt_pool(x)
        x = self.flatten(x)
        x = torch.cat([x, hrv], dim=1)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
 
 
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
    for x, hrv, y in loader:
        x = x.to(device)
        hrv = hrv.to(device)
        logits = model(x, hrv)
        pred = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(pred)
        all_labels.append(y.numpy())
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    cm = _confusion_matrix(y_true, y_pred, len(CLASSES_BIN))
    metrics = _metrics_from_cm(cm)
    metrics["confusion_matrix"] = cm
    return metrics
 
 
def _compute_train_stats(
    df: pd.DataFrame,
    rr_key: str,
    rr_clip_low: float,
    rr_clip_high: float,
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    rr_count = 0
    rr_mean = 0.0
    rr_m2 = 0.0
    delta_count = 0
    delta_mean = 0.0
    delta_m2 = 0.0
    hrv_count = 0
    hrv_mean = np.zeros((4,), dtype=np.float64)
    hrv_m2 = np.zeros((4,), dtype=np.float64)
 
    for path in df["file_path"].tolist():
        with np.load(path, allow_pickle=True) as data:
            if rr_key in data:
                rr_ms = data[rr_key]
            elif "rr_ms" in data:
                rr_ms = data["rr_ms"]
            else:
                continue
        rr_ms = _clip_rr(rr_ms, rr_clip_low, rr_clip_high).astype(np.float64)
        hrv = _hrv_features(rr_ms).astype(np.float64)
        hrv_count += 1
        delta_hrv = hrv - hrv_mean
        hrv_mean += delta_hrv / hrv_count
        hrv_m2 += delta_hrv * (hrv - hrv_mean)
        for v in rr_ms:
            rr_count += 1
            delta = v - rr_mean
            rr_mean += delta / rr_count
            rr_m2 += delta * (v - rr_mean)
        delta_rr = np.diff(rr_ms, prepend=rr_ms[0] if rr_ms.size else 0.0)
        for v in delta_rr:
            delta_count += 1
            delta2 = v - delta_mean
            delta_mean += delta2 / delta_count
            delta_m2 += delta2 * (v - delta_mean)
 
    rr_var = rr_m2 / max(rr_count - 1, 1)
    delta_var = delta_m2 / max(delta_count - 1, 1)
    rr_std = float(np.sqrt(rr_var)) if rr_var > 0 else 1.0
    delta_std = float(np.sqrt(delta_var)) if delta_var > 0 else 1.0
    hrv_var = hrv_m2 / max(hrv_count - 1, 1)
    hrv_std = np.sqrt(np.maximum(hrv_var, 1e-12))
    return float(rr_mean), rr_std, float(delta_mean), delta_std, hrv_mean.astype(np.float32), hrv_std.astype(np.float32)
 
 
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
 
    rr_mean, rr_std, delta_mean, delta_std, hrv_mean, hrv_std = _compute_train_stats(
        train_df, args.rr_key, args.rr_clip_low, args.rr_clip_high
    )
    train_ds = RRBinaryDataset(
        train_df,
        args.seq_len,
        args.rr_key,
        rr_mean,
        rr_std,
        delta_mean,
        delta_std,
        hrv_mean,
        hrv_std,
        args.rr_clip_low,
        args.rr_clip_high,
    )
    val_ds = RRBinaryDataset(
        val_df,
        args.seq_len,
        args.rr_key,
        rr_mean,
        rr_std,
        delta_mean,
        delta_std,
        hrv_mean,
        hrv_std,
        args.rr_clip_low,
        args.rr_clip_high,
    )
    test_ds = RRBinaryDataset(
        test_df,
        args.seq_len,
        args.rr_key,
        rr_mean,
        rr_std,
        delta_mean,
        delta_std,
        hrv_mean,
        hrv_std,
        args.rr_clip_low,
        args.rr_clip_high,
    )
 
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
 
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
 
    hrv_dim = int(hrv_mean.shape[0])
    model = RRLite(2, len(CLASSES_BIN), hrv_dim=hrv_dim, pool_len=args.lite_pool_len).to(device)
 
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
        for x, hrv, y in train_loader:
            x = x.to(device)
            hrv = hrv.to(device)
            y = y.to(device)
            logits = model(x, hrv)
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
                "rr_mean": rr_mean,
                "rr_std": rr_std,
                "delta_mean": delta_mean,
                "delta_std": delta_std,
                "rr_clip_low": args.rr_clip_low,
                "rr_clip_high": args.rr_clip_high,
                "hrv_mean": hrv_mean,
                "hrv_std": hrv_std,
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
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", type=str, choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--save_path", type=str, default="/root/project/ECG/ptbxl_rr_dataset_builder/checkpoints/rr_cnn_binary_lite.pt"
    )
    parser.add_argument("--rr_clip_low", type=float, default=300.0)
    parser.add_argument("--rr_clip_high", type=float, default=2000.0)
    return parser
 
 
if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
