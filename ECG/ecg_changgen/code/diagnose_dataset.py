"""
数据集质量诊断脚本
用于分析ECG数据集的质量问题
"""
import numpy as np
import glob
import os
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_dataset(data_dir):
    """分析数据集的详细统计信息"""
    print("="*80)
    print("数据集质量诊断报告")
    print("="*80)

    # 1. 加载所有数据
    x_files = sorted(glob.glob(os.path.join(data_dir, "X_*.npy")))
    y_files = sorted(glob.glob(os.path.join(data_dir, "Y_*.npy")))

    print(f"\n找到 {len(x_files)} 个 X 文件, {len(y_files)} 个 Y 文件")

    all_X = []
    all_Y = []

    for xf, yf in zip(x_files, y_files):
        X = np.load(xf)
        Y = np.load(yf)
        all_X.append(X)
        all_Y.append(Y)
        print(f"  {os.path.basename(xf)}: {X.shape}, {os.path.basename(yf)}: {Y.shape}")

    X_all = np.concatenate(all_X, axis=0)
    Y_all = np.concatenate(all_Y, axis=0)

    print(f"\n总样本数: {len(Y_all)}")
    print(f"X shape: {X_all.shape}")
    print(f"Y shape: {Y_all.shape}")

    # 2. 类别分布
    print("\n" + "="*80)
    print("类别分布分析")
    print("="*80)
    unique, counts = np.unique(Y_all, return_counts=True)
    total = len(Y_all)

    for cls, count in zip(unique, counts):
        pct = count / total * 100
        print(f"Class {cls}: {count:>8} 样本 ({pct:>6.2f}%)")

    imbalance_ratio = counts.max() / counts.min()
    print(f"\n不平衡比例: {imbalance_ratio:.1f}:1")

    # 3. 数据质量检查
    print("\n" + "="*80)
    print("数据质量检查")
    print("="*80)

    # 3.1 检查NaN和Inf
    nan_count = np.sum(np.isnan(X_all))
    inf_count = np.sum(np.isinf(X_all))
    print(f"NaN 值数量: {nan_count}")
    print(f"Inf 值数量: {inf_count}")

    # 3.2 检查全零样本
    zero_samples = np.sum(np.all(X_all == 0, axis=(1, 2)) if X_all.ndim == 3 else np.all(X_all == 0, axis=1))
    print(f"全零样本数量: {zero_samples} ({zero_samples/len(X_all)*100:.2f}%)")

    # 3.3 数据范围
    print(f"\n数据值范围:")
    print(f"  Min: {np.min(X_all):.6f}")
    print(f"  Max: {np.max(X_all):.6f}")
    print(f"  Mean: {np.mean(X_all):.6f}")
    print(f"  Std: {np.std(X_all):.6f}")

    # 4. 按类别分析数据特征
    print("\n" + "="*80)
    print("各类别数据特征分析")
    print("="*80)

    for cls in unique:
        mask = Y_all == cls
        X_cls = X_all[mask]

        print(f"\nClass {cls}:")
        print(f"  样本数: {len(X_cls)}")
        print(f"  Mean: {np.mean(X_cls):.6f}")
        print(f"  Std: {np.std(X_cls):.6f}")
        print(f"  Min: {np.min(X_cls):.6f}")
        print(f"  Max: {np.max(X_cls):.6f}")

        # 计算每个样本的统计量
        if X_cls.ndim == 3:
            sample_means = np.mean(X_cls, axis=(1, 2))
            sample_stds = np.std(X_cls, axis=(1, 2))
        else:
            sample_means = np.mean(X_cls, axis=1)
            sample_stds = np.std(X_cls, axis=1)

        print(f"  样本均值的均值: {np.mean(sample_means):.6f}")
        print(f"  样本标准差的均值: {np.mean(sample_stds):.6f}")

        # 检查异常样本
        zero_in_cls = np.sum(sample_stds < 1e-6)
        if zero_in_cls > 0:
            print(f"  警告: 发现 {zero_in_cls} 个近乎恒定的样本")

    # 5. 类别间相似性分析
    print("\n" + "="*80)
    print("类别间分布重叠分析")
    print("="*80)

    # 计算每个样本的特征统计量
    if X_all.ndim == 3:
        sample_features = np.column_stack([
            np.mean(X_all, axis=(1, 2)),
            np.std(X_all, axis=(1, 2)),
            np.max(X_all, axis=(1, 2)),
            np.min(X_all, axis=(1, 2)),
        ])
    else:
        sample_features = np.column_stack([
            np.mean(X_all, axis=1),
            np.std(X_all, axis=1),
            np.max(X_all, axis=1),
            np.min(X_all, axis=1),
        ])

    print("\n各类别特征分布:")
    feature_names = ["Mean", "Std", "Max", "Min"]

    for feat_idx, feat_name in enumerate(feature_names):
        print(f"\n{feat_name}:")
        for cls in unique:
            mask = Y_all == cls
            feat_vals = sample_features[mask, feat_idx]
            print(f"  Class {cls}: mean={np.mean(feat_vals):.6f}, std={np.std(feat_vals):.6f}")

    # 6. 推荐的改进措施
    print("\n" + "="*80)
    print("诊断结论与建议")
    print("="*80)

    issues = []
    recommendations = []

    # 检查不平衡问题
    if imbalance_ratio > 100:
        issues.append(f"严重的类别不平衡 ({imbalance_ratio:.1f}:1)")
        recommendations.append("建议:")
        recommendations.append("  1. 收集更多少数类样本（至少每类10,000+样本）")
        recommendations.append("  2. 检查标注是否过于保守（很多S/V被标记为N）")
        recommendations.append("  3. 考虑使用SMOTE等过采样技术")

    # 检查样本数量
    min_samples = counts.min()
    if min_samples < 5000:
        issues.append(f"少数类样本过少 (最少的类只有 {min_samples} 个样本)")
        recommendations.append("  4. 当前S和V类样本数量不足以训练稳健模型")

    # 检查数据质量
    if zero_samples > 0:
        issues.append(f"存在 {zero_samples} 个全零样本")
        recommendations.append("  5. 检查并清理全零或异常样本")

    if nan_count > 0 or inf_count > 0:
        issues.append("存在NaN或Inf值")
        recommendations.append("  6. 清理或填充缺失值")

    print("\n发现的问题:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

    if recommendations:
        print("\n" + "\n".join(recommendations))

    # 7. 采样策略建议
    print("\n" + "="*80)
    print("采样策略建议")
    print("="*80)

    # 当前策略效果分析
    print("\n当前配置 (RESAMPLE_WEIGHTS=0.85,0.10,0.05):")
    print("  - 这会在每个batch中强制包含大量S和V样本")
    print("  - 但由于S和V样本太少且可能标注有误，模型学到了错误的模式")

    # 建议新的权重
    print("\n建议尝试的配置:")
    print("  1. 降低重采样强度: RESAMPLE_WEIGHTS=0.94,0.03,0.03")
    print("  2. 或使用自动权重: RESAMPLE_WEIGHTS=auto (配合 RESAMPLE_ALPHA=0.3)")
    print("  3. 增加focal loss gamma: FOCAL_GAMMA=3.0 或 4.0")
    print("  4. 调整决策阈值: S_THRESHOLD=0.7, V_THRESHOLD=0.8")

    return X_all, Y_all, unique, counts


def sample_inspection(X_all, Y_all, n_samples=5):
    """抽样检查每个类别的实际样本"""
    print("\n" + "="*80)
    print("样本抽样检查")
    print("="*80)

    unique = np.unique(Y_all)

    for cls in unique:
        print(f"\n{'='*40}")
        print(f"Class {cls} 样本示例:")
        print(f"{'='*40}")

        mask = Y_all == cls
        indices = np.where(mask)[0]

        # 随机抽取样本
        if len(indices) > n_samples:
            sample_indices = np.random.choice(indices, n_samples, replace=False)
        else:
            sample_indices = indices

        for i, idx in enumerate(sample_indices, 1):
            sample = X_all[idx]
            if sample.ndim == 2:
                sample_1d = sample[:, 0] if sample.shape[1] == 1 else sample[0, :]
            else:
                sample_1d = sample.flatten()

            print(f"\n  样本 {i} (索引 {idx}):")
            print(f"    Shape: {sample.shape}")
            print(f"    Mean: {np.mean(sample):.6f}")
            print(f"    Std: {np.std(sample):.6f}")
            print(f"    Min: {np.min(sample):.6f}")
            print(f"    Max: {np.max(sample):.6f}")
            print(f"    前10个值: {sample_1d[:10]}")
            print(f"    后10个值: {sample_1d[-10:]}")


def check_mislabeling(X_all, Y_all):
    """检查可能的错误标注"""
    print("\n" + "="*80)
    print("潜在标注错误检查")
    print("="*80)

    # 计算简单特征
    if X_all.ndim == 3:
        features = np.column_stack([
            np.mean(X_all, axis=(1, 2)),
            np.std(X_all, axis=(1, 2)),
        ])
    else:
        features = np.column_stack([
            np.mean(X_all, axis=1),
            np.std(X_all, axis=1),
        ])

    # 检查每个类别内的异常样本
    unique = np.unique(Y_all)

    for cls in unique:
        mask = Y_all == cls
        cls_features = features[mask]

        if len(cls_features) < 10:
            continue

        # 计算四分位数
        q1 = np.percentile(cls_features, 25, axis=0)
        q3 = np.percentile(cls_features, 75, axis=0)
        iqr = q3 - q1

        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr

        # 找出异常样本
        outliers = np.any((cls_features < lower_bound) | (cls_features > upper_bound), axis=1)
        n_outliers = np.sum(outliers)

        if n_outliers > 0:
            print(f"\nClass {cls}: 发现 {n_outliers} 个统计异常样本 ({n_outliers/len(cls_features)*100:.2f}%)")
            print(f"  这些样本可能标注错误，建议人工复查")


if __name__ == "__main__":
    import sys

    data_dir = "/root/project/ECG/ecg_changgen/train_hospital"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    print(f"分析数据目录: {data_dir}\n")

    # 主分析
    X_all, Y_all, unique, counts = analyze_dataset(data_dir)

    # 样本抽查
    sample_inspection(X_all, Y_all, n_samples=3)

    # 错误标注检查
    check_mislabeling(X_all, Y_all)

    print("\n" + "="*80)
    print("诊断完成")
    print("="*80)
