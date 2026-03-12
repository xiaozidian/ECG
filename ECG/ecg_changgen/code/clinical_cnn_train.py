'''
python /root/project/ECG/ecg_changgen/code/clinical_cnn_train.py \
  --train-data-dir /root/project/ECG/ecg_changgen/train_hospital \
  --epochs 30 \
  --sampling gpu_resample
'''
import argparse
import glob
import os
import sys
from pathlib import Path
import numpy as np

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data-dir",
        default="/root/project/ECG/ecg_changgen/train_data",
    )
    parser.add_argument(
        "--model-save-path",
        default="/root/project/ECG/ecg_changgen/model/clinical_cnn_mitbih_500hz.h5",
    )
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--normalize-input", dest="normalize_input", action="store_true")
    parser.add_argument("--no-normalize-input", dest="normalize_input", action="store_false")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--shuffle-buffer", type=int, default=100000)
    parser.add_argument("--use-mixed-precision", action="store_true")
    parser.add_argument("--use-xla", dest="use_xla", action="store_true")
    parser.add_argument("--no-xla", dest="use_xla", action="store_false")
    parser.add_argument("--steps-per-execution", type=int, default=100)
    parser.add_argument("--override-steps-per-epoch", type=int, default=0)
    parser.add_argument("--override-validation-steps", type=int, default=0)
    parser.add_argument("--sampling", default="gpu_resample")
    parser.add_argument("--resample-weights", default="0.85,0.10,0.05")
    parser.add_argument("--resample-alpha", type=float, default=0.5)
    parser.add_argument("--min-class-fraction", type=float, default=0.01)
    parser.add_argument("--max-class-fraction", type=float, default=0.12)
    parser.add_argument("--loss", default="sce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", default="auto")
    parser.add_argument("--focal-alpha-power", type=float, default=1.0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--aug-noise-std", type=float, default=0.01)
    parser.add_argument("--aug-scale-min", type=float, default=0.9)
    parser.add_argument("--aug-scale-max", type=float, default=1.1)
    parser.add_argument("--aug-shift-max", type=int, default=5)
    parser.add_argument("--decision-rule", default="threshold")
    parser.add_argument("--s-threshold", type=float, default=0.55)
    parser.add_argument("--v-threshold", type=float, default=0.65)
    parser.add_argument("--target-priors", default="train")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--use-class-weight", action="store_true")
    parser.set_defaults(normalize_input=True, use_xla=True)
    args = parser.parse_args()
    return args


ARGS = _parse_args()

TRAIN_DATA_DIR = ARGS.train_data_dir
MODEL_SAVE_PATH = ARGS.model_save_path
NUM_CLASSES = int(ARGS.num_classes)
BATCH_SIZE = int(ARGS.batch_size)
LEARNING_RATE = float(ARGS.learning_rate)
GRAD_CLIP_NORM = float(ARGS.grad_clip_norm)
NORMALIZE_INPUT = bool(ARGS.normalize_input)
MAX_SAMPLES = int(ARGS.max_samples)
EPOCHS = int(ARGS.epochs)
VAL_SPLIT = float(ARGS.val_split)
SHUFFLE_BUFFER = int(ARGS.shuffle_buffer)
USE_MIXED_PRECISION = bool(ARGS.use_mixed_precision)
USE_XLA = bool(ARGS.use_xla)
STEPS_PER_EXECUTION = int(ARGS.steps_per_execution)
OVERRIDE_STEPS_PER_EPOCH = int(ARGS.override_steps_per_epoch)
OVERRIDE_VALIDATION_STEPS = int(ARGS.override_validation_steps)
SAMPLING = str(ARGS.sampling)
RESAMPLE_WEIGHTS = str(ARGS.resample_weights)
RESAMPLE_ALPHA = float(ARGS.resample_alpha)
MIN_CLASS_FRACTION = float(ARGS.min_class_fraction)
MAX_CLASS_FRACTION = float(ARGS.max_class_fraction)
LOSS_NAME = str(ARGS.loss).strip().lower()
FOCAL_GAMMA = float(ARGS.focal_gamma)
FOCAL_ALPHA = str(ARGS.focal_alpha).strip().lower()
FOCAL_ALPHA_POWER = float(ARGS.focal_alpha_power)
AUGMENT = bool(ARGS.augment)
AUG_NOISE_STD = float(ARGS.aug_noise_std)
AUG_SCALE_MIN = float(ARGS.aug_scale_min)
AUG_SCALE_MAX = float(ARGS.aug_scale_max)
AUG_SHIFT_MAX = int(ARGS.aug_shift_max)
DECISION_RULE = str(ARGS.decision_rule).strip().lower()
S_THRESHOLD = float(ARGS.s_threshold)
V_THRESHOLD = float(ARGS.v_threshold)
TARGET_PRIORS = str(ARGS.target_priors).strip()
EVAL_ONLY = bool(ARGS.eval_only)
USE_CLASS_WEIGHT = bool(ARGS.use_class_weight)
if USE_CLASS_WEIGHT and str(SAMPLING).lower() in ("resample", "gpu_resample"):
    print("USE_CLASS_WEIGHT disabled because resample is enabled.")
    USE_CLASS_WEIGHT = False


def _stratified_train_test_split(X, y, test_size=0.2, seed=42, num_classes=3):
    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []
    for cls in range(num_classes):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_test = int(round(len(idx) * float(test_size)))
        if n_test <= 0:
            train_indices.append(idx)
            continue
        test_indices.append(idx[:n_test])
        train_indices.append(idx[n_test:])

    train_idx = (
        np.concatenate(train_indices)
        if train_indices
        else np.array([], dtype=np.int64)
    )
    test_idx = (
        np.concatenate(test_indices)
        if test_indices
        else np.array([], dtype=np.int64)
    )
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def _classification_report(y_true, y_pred, num_classes=3, digits=4):
    eps = 1e-12
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)

    lines = []
    header = (
        f"{'class':>9} {'precision':>9} {'recall':>9} "
        f"{'f1-score':>9} {'support':>9}"
    )
    lines.append(header)

    precisions = []
    recalls = []
    f1s = []
    supports = []

    for cls in range(num_classes):
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        support = int(np.sum(y_true == cls))
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        row = (
            f"{cls:>9} {precision:>9.{digits}f} {recall:>9.{digits}f} "
            f"{f1:>9.{digits}f} {support:>9d}"
        )
        lines.append(row)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    total_support = float(np.sum(supports)) + eps
    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    weights = (
        np.array(supports, dtype=np.float64) / total_support
        if supports
        else np.array([], dtype=np.float64)
    )
    weighted_precision = (
        float(np.sum(np.array(precisions) * weights)) if precisions else 0.0
    )
    weighted_recall = (
        float(np.sum(np.array(recalls) * weights)) if recalls else 0.0
    )
    weighted_f1 = float(np.sum(np.array(f1s) * weights)) if f1s else 0.0

    lines.append("")
    macro_row = (
        f"{'macro avg':>9} {macro_precision:>9.{digits}f} "
        f"{macro_recall:>9.{digits}f} {macro_f1:>9.{digits}f} "
        f"{int(total_support):>9d}"
    )
    weighted_row = (
        f"{'weighted avg':>9} {weighted_precision:>9.{digits}f} "
        f"{weighted_recall:>9.{digits}f} {weighted_f1:>9.{digits}f} "
        f"{int(total_support):>9d}"
    )
    lines.append(macro_row)
    lines.append(weighted_row)
    return "\n".join(lines)


def _prepend_library_paths(paths):
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    existing_parts = [p for p in existing.split(":") if p]
    to_add = []
    for p in paths:
        if not p:
            continue
        p = str(p)
        if p in existing_parts:
            continue
        if os.path.isdir(p):
            to_add.append(p)
    if not to_add:
        return False
    os.environ["LD_LIBRARY_PATH"] = ":".join(to_add + existing_parts)
    return True


def _configure_tensorflow():
    cuda_candidate_dirs = [
        "/usr/local/cuda-11.8/targets/x86_64-linux/lib",
        "/usr/local/cuda-11.8/lib64",
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
    ]

    nvidia_candidate_dirs = []
    try:
        import nvidia.cudnn

        nvidia_candidate_dirs.append(
            str(Path(nvidia.cudnn.__file__).resolve().parent / "lib")
        )
    except Exception:
        pass

    try:
        import nvidia.cublas

        nvidia_candidate_dirs.append(
            str(Path(nvidia.cublas.__file__).resolve().parent / "lib")
        )
    except Exception:
        pass

    candidate_dirs = nvidia_candidate_dirs + cuda_candidate_dirs
    changed = _prepend_library_paths(candidate_dirs)
    if (
        changed
        and globals().get("__name__") == "__main__"
        and os.environ.get("ECG_TF_REEXEC", "0") != "1"
        and sys.argv
        and os.path.isfile(sys.argv[0])
    ):
        os.environ["ECG_TF_REEXEC"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    import tensorflow as tf

    if USE_MIXED_PRECISION:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("Warning: TensorFlow did not detect GPU.")
    else:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

    print(f"Python: {sys.executable}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"Visible GPUs: {gpus}")
    return tf


def load_all_data_efficiently(train_dir):
    """
    Loads all data into a pre-allocated numpy array to optimize memory usage
    and allow for full dataset shuffling and GPU pre-loading.
    """
    print("Scanning data files...")
    x_files = glob.glob(os.path.join(train_dir, "X_*.npy"))
    y_files = glob.glob(os.path.join(train_dir, "Y_*.npy"))

    if not x_files:
        raise FileNotFoundError("No X_*.npy files found.")

    def _file_key(path, prefix):
        base = os.path.basename(path)
        if not base.startswith(prefix) or not base.endswith(".npy"):
            return None
        return base[len(prefix) : -4]

    x_map = {}
    for p in x_files:
        k = _file_key(p, "X_")
        if k is not None:
            x_map[k] = p
    y_map = {}
    for p in y_files:
        k = _file_key(p, "Y_")
        if k is not None:
            y_map[k] = p

    common_keys = sorted(
        set(x_map.keys()) & set(y_map.keys()),
        key=lambda k: int(k) if str(k).isdigit() else str(k),
    )
    if not common_keys:
        raise FileNotFoundError("No paired X_*.npy / Y_*.npy files found.")
    if len(common_keys) != len(x_map) or len(common_keys) != len(y_map):
        missing_x = sorted(set(y_map.keys()) - set(x_map.keys()))
        missing_y = sorted(set(x_map.keys()) - set(y_map.keys()))
        print(
            f"Warning: Unpaired files. "
            f"missing_x={len(missing_x)}, missing_y={len(missing_y)}"
        )

    # 1. Calculate total size first to pre-allocate
    total_samples = 0
    print("Calculating total dataset size...")
    for k in common_keys:
        yf = y_map[k]
        # mmap_mode='r' allows reading shape without loading data
        y_mmap = np.load(yf, mmap_mode='r')
        total_samples += len(y_mmap)

    if MAX_SAMPLES > 0:
        total_samples = min(int(total_samples), int(MAX_SAMPLES))

    print(f"Total samples found: {total_samples}")

    # 2. Pre-allocate memory
    # 500Hz data length is 748. Using float32 saves 50% RAM vs float64.
    # 954k * 748 * 4 bytes approx 2.8 GB -> Fits easily in RAM.
    print("Pre-allocating memory...")
    X_all = np.zeros((total_samples, 748, 1), dtype=np.float32)
    Y_all = np.zeros((total_samples,), dtype=np.int8)

    # 3. Load data into pre-allocated arrays
    start_idx = 0
    nonfinite_fixed = 0
    for k in common_keys:
        if start_idx >= total_samples:
            break
        xf = x_map[k]
        yf = y_map[k]
        print(f"Loading {os.path.basename(xf)}...")

        # Load file content
        x_data = np.load(xf).astype(np.float32, copy=False)
        y_data = np.load(yf).astype(np.int8, copy=False)

        # Ensure correct shape
        if x_data.ndim == 2:
            x_data = x_data[..., np.newaxis]

        if len(x_data) != len(y_data):
            raise ValueError(
                f"X/Y length mismatch for key={k}: "
                f"{os.path.basename(xf)} has {len(x_data)}, "
                f"{os.path.basename(yf)} has {len(y_data)}"
            )

        y_min = int(np.min(y_data)) if len(y_data) else 0
        y_max = int(np.max(y_data)) if len(y_data) else 0
        if y_min < 0 or y_max >= NUM_CLASSES:
            uniq = np.unique(y_data)
            head = uniq[:20].tolist()
            raise ValueError(
                f"Invalid labels in {os.path.basename(yf)}: "
                f"min={y_min}, max={y_max}, unique[:20]={head}"
            )

        bad = int(np.count_nonzero(~np.isfinite(x_data)))
        if bad:
            nonfinite_fixed += bad
            x_data = np.nan_to_num(
                x_data,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32, copy=False)

        n_samples = len(x_data)
        n_to_copy = min(int(n_samples), int(total_samples - start_idx))
        end_idx = start_idx + n_to_copy

        # Fill slice
        X_all[start_idx:end_idx] = x_data[:n_to_copy]
        Y_all[start_idx:end_idx] = y_data[:n_to_copy]

        start_idx = end_idx

        # Explicitly delete to free temp memory
        del x_data, y_data

    print("All data loaded into memory.")
    if nonfinite_fixed:
        print(f"Warning: fixed non-finite X values: {nonfinite_fixed}")
    return X_all, Y_all


def _make_sparse_focal_loss(tf, gamma, alpha_vec=None):
    eps = tf.constant(1e-7, dtype=tf.float32)
    gamma = tf.constant(float(gamma), dtype=tf.float32)
    alpha = (
        tf.constant(alpha_vec, dtype=tf.float32) if alpha_vec is not None else None
    )

    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        y_onehot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1], dtype=tf.float32)
        p_t = tf.reduce_sum(y_pred * y_onehot, axis=-1)
        ce = -tf.math.log(p_t)
        mod = tf.pow(1.0 - p_t, gamma)
        if alpha is not None:
            a_t = tf.gather(alpha, y_true)
            return tf.reduce_mean(a_t * mod * ce)
        return tf.reduce_mean(mod * ce)

    return loss


def _parse_alpha_list(text, num_classes):
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != num_classes:
        raise ValueError(
            f"FOCAL_ALPHA must have {num_classes} values, got: {text}"
        )
    vals = np.array([float(p) for p in parts], dtype=np.float64)
    s = float(np.sum(vals))
    if s <= 0:
        raise ValueError(f"FOCAL_ALPHA sum must be > 0, got: {vals.tolist()}")
    vals = vals / s
    return vals.astype(np.float32).tolist()


def _parse_prob_list(text, num_classes, name):
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != num_classes:
        raise ValueError(f"{name} must have {num_classes} values, got: {text}")
    vals = np.array([float(p) for p in parts], dtype=np.float64)
    s = float(np.sum(vals))
    if s <= 0:
        raise ValueError(f"{name} sum must be > 0, got: {vals.tolist()}")
    vals = vals / s
    return vals.astype(np.float32).tolist()


def get_model(tf, class_priors=None):
    nclass = NUM_CLASSES
    # 500Hz, window 748
    inp = tf.keras.layers.Input(shape=(748, 1))

    # Block 1
    img_1 = tf.keras.layers.Convolution1D(
        16,
        kernel_size=15,
        activation=tf.keras.activations.relu,
        padding="same",
    )(inp)
    img_1 = tf.keras.layers.Convolution1D(
        16,
        kernel_size=15,
        activation=tf.keras.activations.relu,
        padding="same",
    )(img_1)
    img_1 = tf.keras.layers.MaxPool1D(pool_size=2)(img_1)
    img_1 = tf.keras.layers.Dropout(rate=0.1)(img_1)

    # Block 2
    img_1 = tf.keras.layers.Convolution1D(
        32,
        kernel_size=9,
        activation=tf.keras.activations.relu,
        padding="same",
    )(img_1)
    img_1 = tf.keras.layers.Convolution1D(
        32,
        kernel_size=9,
        activation=tf.keras.activations.relu,
        padding="same",
    )(img_1)
    img_1 = tf.keras.layers.MaxPool1D(pool_size=2)(img_1)
    img_1 = tf.keras.layers.Dropout(rate=0.1)(img_1)

    # Block 3
    img_1 = tf.keras.layers.Convolution1D(
        64,
        kernel_size=5,
        activation=tf.keras.activations.relu,
        padding="same",
    )(img_1)
    img_1 = tf.keras.layers.Convolution1D(
        64,
        kernel_size=5,
        activation=tf.keras.activations.relu,
        padding="same",
    )(img_1)
    img_1 = tf.keras.layers.MaxPool1D(pool_size=2)(img_1)
    img_1 = tf.keras.layers.Dropout(rate=0.1)(img_1)

    # Block 4
    img_1 = tf.keras.layers.Convolution1D(
        256,
        kernel_size=3,
        activation=tf.keras.activations.relu,
        padding="same",
    )(img_1)
    img_1 = tf.keras.layers.Convolution1D(
        256,
        kernel_size=3,
        activation=tf.keras.activations.relu,
        padding="same",
    )(img_1)
    img_1 = tf.keras.layers.GlobalMaxPool1D()(img_1)
    img_1 = tf.keras.layers.Dropout(rate=0.2)(img_1)

    # Dense layers
    dense_1 = tf.keras.layers.Dense(
        64,
        activation=tf.keras.activations.relu,
        name="dense_1",
    )(img_1)
    dense_1 = tf.keras.layers.Dense(
        64,
        activation=tf.keras.activations.relu,
        name="dense_2",
    )(dense_1)
    dense_1 = tf.keras.layers.Dense(
        nclass,
        activation=tf.keras.activations.softmax,
        name="dense_3_mitbih",
        dtype="float32",
    )(dense_1)

    model = tf.keras.models.Model(inputs=inp, outputs=dense_1)
    opt_kwargs = {"learning_rate": LEARNING_RATE}
    if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
        opt_kwargs["clipnorm"] = GRAD_CLIP_NORM
    opt = tf.keras.optimizers.Adam(**opt_kwargs)

    loss_obj = tf.keras.losses.sparse_categorical_crossentropy
    if LOSS_NAME == "focal":
        alpha_vec = None
        if FOCAL_ALPHA not in ("", "none", "null", "false", "0"):
            if FOCAL_ALPHA == "auto":
                pri = (
                    np.asarray(class_priors, dtype=np.float64)
                    if class_priors is not None
                    else None
                )
                if pri is None or pri.size != nclass:
                    pri = np.ones((nclass,), dtype=np.float64) / float(nclass)
                eps = 1e-12
                inv = 1.0 / np.maximum(pri, eps)
                inv = np.power(inv, float(FOCAL_ALPHA_POWER))
                alpha_vec = (inv / np.sum(inv)).astype(np.float32).tolist()
            else:
                alpha_vec = _parse_alpha_list(FOCAL_ALPHA, nclass)
        loss_obj = _make_sparse_focal_loss(tf, gamma=FOCAL_GAMMA, alpha_vec=alpha_vec)

    model.compile(
        optimizer=opt,
        loss=loss_obj,
        metrics=["acc"],
        jit_compile=USE_XLA,
        steps_per_execution=STEPS_PER_EXECUTION,
    )
    model.summary()
    return model


def main():
    if os.environ.get("SMOKE_TEST", "0") == "1":
        tf = _configure_tensorflow()
        model = get_model(tf)
        steps = (
            OVERRIDE_STEPS_PER_EPOCH
            if OVERRIDE_STEPS_PER_EPOCH > 0
            else 10
        )
        x = tf.random.uniform([BATCH_SIZE, 748, 1], dtype=tf.float32)
        y = tf.random.uniform(
            [BATCH_SIZE],
            minval=0,
            maxval=NUM_CLASSES,
            dtype=tf.int32,
        )
        ds = tf.data.Dataset.from_tensors((x, y)).repeat()
        ds = ds.prefetch(tf.data.AUTOTUNE)
        model.fit(ds, steps_per_epoch=steps, epochs=1, verbose=1)
        return

    print(
        "Config: "
        f"EPOCHS={EPOCHS}, "
        f"BATCH_SIZE={BATCH_SIZE}, "
        f"SAMPLING={SAMPLING}, "
        f"RESAMPLE_WEIGHTS={RESAMPLE_WEIGHTS}, "
        f"RESAMPLE_ALPHA={RESAMPLE_ALPHA}, "
        f"MIN_CLASS_FRACTION={MIN_CLASS_FRACTION}, "
        f"MAX_CLASS_FRACTION={MAX_CLASS_FRACTION}, "
        f"LOSS={LOSS_NAME}, "
        f"FOCAL_GAMMA={FOCAL_GAMMA}, "
        f"FOCAL_ALPHA={FOCAL_ALPHA}, "
        f"FOCAL_ALPHA_POWER={FOCAL_ALPHA_POWER}, "
        f"AUGMENT={int(AUGMENT)}, "
        f"AUG_NOISE_STD={AUG_NOISE_STD}, "
        f"AUG_SCALE=[{AUG_SCALE_MIN},{AUG_SCALE_MAX}], "
        f"AUG_SHIFT_MAX={AUG_SHIFT_MAX}, "
        f"DECISION_RULE={DECISION_RULE}, "
        f"S_THRESHOLD={S_THRESHOLD}, "
        f"V_THRESHOLD={V_THRESHOLD}, "
        f"TARGET_PRIORS={TARGET_PRIORS}, "
        f"EVAL_ONLY={int(EVAL_ONLY)}, "
        f"USE_CLASS_WEIGHT={int(USE_CLASS_WEIGHT)}, "
        f"USE_XLA={int(USE_XLA)}, "
        f"USE_MIXED_PRECISION={int(USE_MIXED_PRECISION)}, "
        f"STEPS_PER_EXECUTION={STEPS_PER_EXECUTION}, "
        f"OVERRIDE_STEPS_PER_EPOCH={OVERRIDE_STEPS_PER_EPOCH}, "
        f"OVERRIDE_VALIDATION_STEPS={OVERRIDE_VALIDATION_STEPS}"
    )
    if EPOCHS <= 2:
        print(
            "Warning: EPOCHS<=2. "
            "如果不是在做快速验证，请把 EPOCHS 调大（例如 30~50）。"
        )

    tf = _configure_tensorflow()

    # 1. Load Data (Into RAM)
    X, Y = load_all_data_efficiently(TRAIN_DATA_DIR)

    # 2. Split
    print("Splitting data...")
    X_train, X_test, Y_train, Y_test = _stratified_train_test_split(
        X, Y, test_size=0.2, seed=42, num_classes=NUM_CLASSES
    )

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    # Free original arrays
    del X, Y

    # 3. Handle Imbalance (Class Weights)
    unique_classes, counts = np.unique(Y_train, return_counts=True)
    total_y = len(Y_train)
    class_weights = {
        cls: total_y / (len(unique_classes) * count)
        for cls, count in zip(unique_classes, counts)
    }
    print(f"Computed Class Weights: {class_weights}")
    class_counts = {
        int(cls): int(count)
        for cls, count in zip(unique_classes, counts)
    }
    class_priors = np.zeros((NUM_CLASSES,), dtype=np.float32)
    for cls, count in zip(unique_classes, counts):
        if 0 <= int(cls) < NUM_CLASSES:
            class_priors[int(cls)] = float(count) / float(total_y)
    print(f"Class Priors: {class_priors.tolist()}")

    # 4. Create tf.data.Dataset (Optimized for GPU)
    print("Moving data to GPU memory...")

    # Move numpy arrays to GPU tensors explicitly
    # This ensures the entire dataset resides in GPU VRAM
    gpus = tf.config.list_physical_devices("GPU")
    device = "/device:GPU:0" if gpus else "/device:CPU:0"
    with tf.device(device):
        X_train_gpu = tf.convert_to_tensor(X_train, dtype=tf.float32)
        Y_train_gpu = tf.convert_to_tensor(Y_train, dtype=tf.int32)
        X_test_gpu = tf.convert_to_tensor(X_test, dtype=tf.float32)
        Y_test_gpu = tf.convert_to_tensor(Y_test, dtype=tf.int32)

    del X_train, X_test, Y_train
    import gc
    gc.collect()

    print("Creating TensorFlow Datasets from GPU tensors...")

    def _preprocess_base(x):
        x = tf.cast(x, tf.float32)
        x = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))
        if NORMALIZE_INPUT:
            mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
            var = tf.reduce_mean(
                tf.square(x - mean), axis=[1, 2], keepdims=True
            )
            denom = tf.maximum(
                tf.sqrt(var), tf.constant(1e-6, dtype=tf.float32)
            )
            x = (x - mean) / denom
        return x

    def _augment_x(x):
        if AUG_SHIFT_MAX and AUG_SHIFT_MAX > 0:
            shift = tf.random.uniform(
                [], -AUG_SHIFT_MAX, AUG_SHIFT_MAX + 1, dtype=tf.int32
            )
            x = tf.roll(x, shift=shift, axis=1)
        if AUG_SCALE_MIN != 1.0 or AUG_SCALE_MAX != 1.0:
            scale = tf.random.uniform(
                [], AUG_SCALE_MIN, AUG_SCALE_MAX, dtype=tf.float32
            )
            x = x * scale
        if AUG_NOISE_STD and AUG_NOISE_STD > 0:
            noise = tf.random.normal(tf.shape(x), stddev=AUG_NOISE_STD)
            x = x + tf.cast(noise, x.dtype)
        x = tf.clip_by_value(x, 0.0, 1.0)
        return x

    def _preprocess_train(x, y):
        x = _preprocess_base(x)
        if AUGMENT:
            x = _augment_x(x)
        y = tf.cast(y, tf.int32)
        return x, y

    def _preprocess_eval(x, y):
        x = _preprocess_base(x)
        y = tf.cast(y, tf.int32)
        return x, y

    train_size = int(X_train_gpu.shape[0])
    steps_per_epoch = int(np.ceil(train_size / float(BATCH_SIZE)))
    if OVERRIDE_STEPS_PER_EPOCH > 0:
        steps_per_epoch = OVERRIDE_STEPS_PER_EPOCH

    if SAMPLING.lower() == "gpu_resample":
        if RESAMPLE_WEIGHTS.strip().lower() == "auto":
            pri = class_priors.astype(np.float64)
            eps = 1e-12
            inv = 1.0 / np.maximum(pri, eps)
            inv = np.power(inv, float(RESAMPLE_ALPHA))
            weights = (inv / np.sum(inv)).astype(np.float64).tolist()
        else:
            weights = [
                w.strip() for w in RESAMPLE_WEIGHTS.split(",") if w.strip()
            ]
            if len(weights) != NUM_CLASSES:
                raise ValueError(
                    f"RESAMPLE_WEIGHTS must have {NUM_CLASSES} values, got: "
                    f"{RESAMPLE_WEIGHTS}"
                )
            weights = [float(w) for w in weights]
            s = float(np.sum(weights))
            if s <= 0:
                raise ValueError(
                    f"RESAMPLE_WEIGHTS sum must be > 0, got: {weights}"
                )
            weights = (np.array(weights, dtype=np.float64) / s).tolist()

        min_count = int(round(float(MIN_CLASS_FRACTION) * float(BATCH_SIZE)))
        min_count = max(0, min_count)
        max_count = int(round(float(MAX_CLASS_FRACTION) * float(BATCH_SIZE)))
        max_count = max(0, max_count)
        if max_count and min_count > max_count:
            min_count = max_count

        n1 = int(round(float(weights[1]) * float(BATCH_SIZE)))
        n2 = int(round(float(weights[2]) * float(BATCH_SIZE)))
        n1 = max(min_count, n1)
        n2 = max(min_count, n2)
        if max_count:
            n1 = min(max_count, n1)
            n2 = min(max_count, n2)
        n0 = int(BATCH_SIZE) - int(n1) - int(n2)
        if n0 < 1:
            overflow = 1 - int(n0)
            d1 = min(int(overflow), max(0, int(n1) - min_count))
            n1 = int(n1) - int(d1)
            overflow = int(overflow) - int(d1)
            d2 = min(int(overflow), max(0, int(n2) - min_count))
            n2 = int(n2) - int(d2)
            n0 = int(BATCH_SIZE) - int(n1) - int(n2)
        if n0 < 1:
            raise ValueError(
                f"Invalid batch class counts: n0={n0}, n1={n1}, n2={n2}, "
                f"BATCH_SIZE={BATCH_SIZE}, weights={weights}, "
                f"MIN_CLASS_FRACTION={MIN_CLASS_FRACTION}"
            )

        idx0 = tf.reshape(tf.where(tf.equal(Y_train_gpu, 0)), (-1,))
        idx1 = tf.reshape(tf.where(tf.equal(Y_train_gpu, 1)), (-1,))
        idx2 = tf.reshape(tf.where(tf.equal(Y_train_gpu, 2)), (-1,))

        def _sample_batch(_):
            tf.debugging.assert_positive(tf.shape(idx0)[0])
            tf.debugging.assert_positive(tf.shape(idx1)[0])
            tf.debugging.assert_positive(tf.shape(idx2)[0])

            r0 = tf.random.uniform(
                [n0], 0, tf.shape(idx0)[0], dtype=tf.int32
            )
            r1 = tf.random.uniform(
                [n1], 0, tf.shape(idx1)[0], dtype=tf.int32
            )
            r2 = tf.random.uniform(
                [n2], 0, tf.shape(idx2)[0], dtype=tf.int32
            )
            s0 = tf.gather(idx0, r0)
            s1 = tf.gather(idx1, r1)
            s2 = tf.gather(idx2, r2)
            idx = tf.concat([s0, s1, s2], axis=0)
            perm = tf.random.shuffle(tf.range(tf.shape(idx)[0]))
            idx = tf.gather(idx, perm)
            x = tf.gather(X_train_gpu, idx)
            y = tf.gather(Y_train_gpu, idx)
            return x, y

        train_ds = tf.data.Dataset.range(steps_per_epoch).repeat()
        train_ds = train_ds.map(
            _sample_batch,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        train_ds = train_ds.map(
            _preprocess_train,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        print(
            f"Sampling: gpu_resample, weights={weights}, "
            f"batch_counts={[n0, n1, n2]}, "
            f"steps_per_epoch={steps_per_epoch}"
        )
    elif SAMPLING.lower() == "resample":
        weights = [w.strip() for w in RESAMPLE_WEIGHTS.split(",") if w.strip()]
        if len(weights) != NUM_CLASSES:
            raise ValueError(
                f"RESAMPLE_WEIGHTS must have {NUM_CLASSES} values, got: "
                f"{RESAMPLE_WEIGHTS}"
            )
        weights = [float(w) for w in weights]
        base_ds = tf.data.Dataset.from_tensor_slices(
            (X_train_gpu, Y_train_gpu)
        )
        buf0 = min(50000, max(1, class_counts.get(0, 1)))
        buf1 = min(5000, max(1, class_counts.get(1, 1)))
        buf2 = min(10000, max(1, class_counts.get(2, 1)))
        ds0 = base_ds.filter(lambda x, y: tf.equal(y, 0))
        ds0 = ds0.shuffle(buf0).repeat()
        ds1 = base_ds.filter(lambda x, y: tf.equal(y, 1))
        ds1 = ds1.shuffle(buf1).repeat()
        ds2 = base_ds.filter(lambda x, y: tf.equal(y, 2))
        ds2 = ds2.shuffle(buf2).repeat()
        train_ds = tf.data.Dataset.sample_from_datasets(
            [ds0, ds1, ds2],
            weights=weights,
        )
        train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
        train_ds = train_ds.map(
            _preprocess_train,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        print(
            f"Sampling: resample, weights={weights}, "
            f"shuffle_buffers={[buf0, buf1, buf2]}, "
            f"steps_per_epoch={steps_per_epoch}"
        )
    else:
        train_ds = tf.data.Dataset.from_tensor_slices(
            (X_train_gpu, Y_train_gpu)
        )
        train_ds = train_ds.shuffle(50000, reshuffle_each_iteration=True)
        train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
        train_ds = train_ds.map(
            _preprocess_train,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        steps_per_epoch = None
        print("Sampling: native shuffle")

    val_ds = tf.data.Dataset.from_tensor_slices((X_test_gpu, Y_test_gpu))
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.map(
        _preprocess_eval,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # 5. Model
    model = get_model(tf, class_priors=class_priors)

    # Callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
    )
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
        verbose=1,
    )
    redonplat = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        patience=3,
        verbose=2,
    )
    callbacks_list = [
        tf.keras.callbacks.TerminateOnNaN(),
        checkpoint,
        early,
        redonplat,
    ]

    # 6. Train
    print("\nStarting training...")
    fit_kwargs = {}
    if steps_per_epoch is not None:
        fit_kwargs["steps_per_epoch"] = steps_per_epoch
    if OVERRIDE_VALIDATION_STEPS > 0:
        fit_kwargs["validation_steps"] = OVERRIDE_VALIDATION_STEPS
    if USE_CLASS_WEIGHT:
        fit_kwargs["class_weight"] = class_weights

    model.fit(
        train_ds,
        epochs=EPOCHS,
        verbose=1,
        callbacks=callbacks_list,
        validation_data=val_ds,
        **fit_kwargs,
    )

    # 7. Evaluate
    print("\nLoading best weights for evaluation...")
    model.load_weights(MODEL_SAVE_PATH)

    print("Evaluating on test set...")
    loss, acc = model.evaluate(val_ds)
    print(f"\nTest Accuracy: {acc:.4f}")

    # 8. Detailed Report
    print("Generating classification report...")
    y_pred_probs = model.predict(val_ds)
    y_pred = np.argmax(y_pred_probs, axis=-1)

    report = _classification_report(
        Y_test,
        y_pred,
        num_classes=NUM_CLASSES,
        digits=4,
    )
    print(report)

    if DECISION_RULE in ("threshold", "thresholds", "hier", "hierarchical"):
        y_pred_thr = np.zeros((y_pred_probs.shape[0],), dtype=np.int64)
        v_mask = y_pred_probs[:, 2] >= float(V_THRESHOLD)
        s_mask = (~v_mask) & (y_pred_probs[:, 1] >= float(S_THRESHOLD))
        y_pred_thr[v_mask] = 2
        y_pred_thr[s_mask] = 1
        print(
            "\nThresholded classification report: "
            f"S_THRESHOLD={S_THRESHOLD}, V_THRESHOLD={V_THRESHOLD}"
        )
        report_thr = _classification_report(
            Y_test,
            y_pred_thr,
            num_classes=NUM_CLASSES,
            digits=4,
        )
        print(report_thr)

    tgt = str(TARGET_PRIORS).strip()
    tgt_low = tgt.lower()
    if tgt_low not in ("", "none", "off", "false", "0", "train"):
        p_train = np.asarray(class_priors, dtype=np.float32).reshape((-1,))
        p_train = p_train / max(1e-12, float(np.sum(p_train)))
        if tgt_low in ("uniform", "equal"):
            p_target = (
                np.ones((NUM_CLASSES,), dtype=np.float32) / float(NUM_CLASSES)
            )
        elif tgt_low == "legacy_multiply":
            p_target = p_train
            p_train = (
                np.ones((NUM_CLASSES,), dtype=np.float32) / float(NUM_CLASSES)
            )
        else:
            p_target = np.asarray(
                _parse_prob_list(tgt, NUM_CLASSES, "TARGET_PRIORS"),
                dtype=np.float32,
            )
        ratio = p_target / np.maximum(p_train, 1e-12)
        y_pred_adj = np.argmax(y_pred_probs * ratio.reshape((1, -1)), axis=-1)
        print(
            "\nPrior-adjusted classification report: "
            f"TARGET_PRIORS={tgt}, train_priors={p_train.tolist()}"
        )
        report_prior = _classification_report(
            Y_test,
            y_pred_adj,
            num_classes=NUM_CLASSES,
            digits=4,
        )
        print(report_prior)


if __name__ == "__main__":
    main()
