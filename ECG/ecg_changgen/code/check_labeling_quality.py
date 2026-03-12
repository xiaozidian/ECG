"""
检查是否存在标注质量问题
通过统计分析找出可能被误标的样本
"""
import numpy as np
import os

def load_data(data_dir):
    """加载数据"""
    X = np.load(os.path.join(data_dir, "X_0.npy"))
    Y = np.load(os.path.join(data_dir, "Y_0.npy"))

    if X.ndim == 2:
        X = X[..., np.newaxis]

    return X, Y


def extract_simple_features(X):
    """提取简单的统计特征"""
    # 展平到2D (n_samples, 748)
    if X.ndim == 3:
        X_2d = X.squeeze(-1)
    else:
        X_2d = X

    features = {}

    # 基础统计量
    features['mean'] = np.mean(X_2d, axis=1)
    features['std'] = np.std(X_2d, axis=1)
    features['max'] = np.max(X_2d, axis=1)
    features['min'] = np.min(X_2d, axis=1)
    features['range'] = features['max'] - features['min']

    # 变异系数
    features['cv'] = features['std'] / (np.abs(features['mean']) + 1e-8)

    # 峰度和偏度的简单近似
    features['max_abs'] = np.max(np.abs(X_2d), axis=1)

    # 一阶差分的统计量（检测信号变化率）
    diff = np.diff(X_2d, axis=1)
    features['diff_mean'] = np.mean(np.abs(diff), axis=1)
    features['diff_max'] = np.max(np.abs(diff), axis=1)

    return features


def find_suspicious_samples(X, Y, n_per_class=50):
    """找出每个类别中统计上最异常的样本"""
    print("\n" + "="*80)
    print("查找统计异常样本（可能的标注错误）")
    print("="*80)

    features = extract_simple_features(X)
    unique_classes = np.unique(Y)

    suspicious_indices = {}

    for cls in unique_classes:
        print(f"\n{'='*40}")
        print(f"Class {cls}")
        print(f"{'='*40}")

        mask = Y == cls
        cls_indices = np.where(mask)[0]

        if len(cls_indices) < 20:
            print(f"样本太少({len(cls_indices)})，跳过")
            continue

        # 对每个特征计算Z-score
        cls_features = {k: v[mask] for k, v in features.items()}

        # 计算综合异常分数
        anomaly_scores = np.zeros(len(cls_indices))

        for feat_name, feat_values in cls_features.items():
            if len(feat_values) < 2:
                continue

            mean = np.mean(feat_values)
            std = np.std(feat_values)

            if std < 1e-8:
                continue

            z_scores = np.abs((feat_values - mean) / std)
            anomaly_scores += z_scores

        # 找出最异常的样本
        n_show = min(n_per_class, len(cls_indices))
        most_anomalous = np.argsort(anomaly_scores)[-n_show:][::-1]

        suspicious_indices[cls] = cls_indices[most_anomalous]

        print(f"\n最异常的{n_show}个样本 (全局索引):")
        print(f"{'索引':<10} {'异常分数':<12} {'mean':<10} {'std':<10} {'max':<10}")
        print("-"*60)

        for i, idx in enumerate(suspicious_indices[cls][:20]):  # 只显示前20个
            local_idx = np.where(cls_indices == idx)[0][0]
            score = anomaly_scores[local_idx]
            mean_val = features['mean'][idx]
            std_val = features['std'][idx]
            max_val = features['max'][idx]

            print(f"{idx:<10} {score:<12.2f} {mean_val:<10.6f} {std_val:<10.6f} {max_val:<10.6f}")

    return suspicious_indices


def compare_class_features(X, Y):
    """比较不同类别的特征分布"""
    print("\n" + "="*80)
    print("类别间特征分布对比")
    print("="*80)

    features = extract_simple_features(X)
    unique_classes = np.unique(Y)

    feature_names = ['mean', 'std', 'max', 'min', 'range', 'cv', 'diff_mean', 'diff_max']

    for feat_name in feature_names:
        if feat_name not in features:
            continue

        print(f"\n{feat_name}:")
        print(f"{'Class':<10} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Samples':<10}")
        print("-"*70)

        for cls in unique_classes:
            mask = Y == cls
            feat_values = features[feat_name][mask]

            if len(feat_values) == 0:
                continue

            mean = np.mean(feat_values)
            std = np.std(feat_values)
            min_val = np.min(feat_values)
            max_val = np.max(feat_values)
            n_samples = len(feat_values)

            print(f"{cls:<10} {mean:<12.6f} {std:<12.6f} {min_val:<12.6f} {max_val:<12.6f} {n_samples:<10}")

    # 检查类别间是否有重叠
    print("\n" + "="*80)
    print("可分性分析")
    print("="*80)

    separability_scores = {}

    for feat_name in feature_names[:6]:  # 只用主要特征
        if feat_name not in features:
            continue

        # 计算类别间距离 (简单的均值差异)
        class_means = {}
        class_stds = {}

        for cls in unique_classes:
            mask = Y == cls
            feat_values = features[feat_name][mask]
            if len(feat_values) > 0:
                class_means[cls] = np.mean(feat_values)
                class_stds[cls] = np.std(feat_values)

        if len(class_means) == 3:
            # 计算Fisher判别比 (简化版)
            between_var = np.var(list(class_means.values()))
            within_var = np.mean(list(class_stds.values())) ** 2

            if within_var > 1e-8:
                fisher_ratio = between_var / within_var
                separability_scores[feat_name] = fisher_ratio

    print("\n特征可分性得分 (Fisher比率, 越大越好):")
    for feat_name, score in sorted(separability_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat_name:<15}: {score:.6f}")

    if separability_scores:
        max_score = max(separability_scores.values())
        if max_score < 0.1:
            print("\n⚠️ 警告: 所有特征的可分性都很低！")
            print("   这表明三个类别在统计特征上高度重叠")
            print("   可能原因:")
            print("   1. 标注质量问题（类别定义不清晰）")
            print("   2. 需要更复杂的特征（如频域特征、形态学特征）")
            print("   3. 数据本身难以区分")


def check_class_overlap(X, Y):
    """检查类别重叠情况"""
    print("\n" + "="*80)
    print("类别重叠检查")
    print("="*80)

    features = extract_simple_features(X)
    unique_classes = np.unique(Y)

    # 使用mean和std作为主要特征
    feat_2d = np.column_stack([features['mean'], features['std']])

    # 对于每个N类样本，检查它是否更接近S或V类的分布
    n_mask = Y == 0
    s_mask = Y == 1
    v_mask = Y == 2

    if not (np.any(n_mask) and np.any(s_mask) and np.any(v_mask)):
        print("缺少某些类别，无法进行重叠检查")
        return

    n_samples = feat_2d[n_mask]
    s_samples = feat_2d[s_mask]
    v_samples = feat_2d[v_mask]

    # 计算每个类别的中心
    n_center = np.mean(n_samples, axis=0)
    s_center = np.mean(s_samples, axis=0)
    v_center = np.mean(v_samples, axis=0)

    print(f"\n类别中心 (mean, std):")
    print(f"  N: ({n_center[0]:.6f}, {n_center[1]:.6f})")
    print(f"  S: ({s_center[0]:.6f}, {s_center[1]:.6f})")
    print(f"  V: ({v_center[0]:.6f}, {v_center[1]:.6f})")

    # 检查有多少N类样本更接近S或V
    n_indices = np.where(n_mask)[0]

    dist_to_n = np.linalg.norm(n_samples - n_center, axis=1)
    dist_to_s = np.linalg.norm(n_samples - s_center, axis=1)
    dist_to_v = np.linalg.norm(n_samples - v_center, axis=1)

    closer_to_s = dist_to_s < dist_to_n
    closer_to_v = dist_to_v < dist_to_n

    print(f"\nN类样本中:")
    print(f"  更接近S类中心的: {np.sum(closer_to_s)} ({np.sum(closer_to_s)/len(n_samples)*100:.2f}%)")
    print(f"  更接近V类中心的: {np.sum(closer_to_v)} ({np.sum(closer_to_v)/len(n_samples)*100:.2f}%)")

    if np.sum(closer_to_s) > 0 or np.sum(closer_to_v) > 0:
        print("\n⚠️ 建议: 人工复查这些可能被误标的N类样本")

        # 保存可疑样本索引
        suspicious_n_indices = n_indices[closer_to_s | closer_to_v]
        print(f"\n可疑N类样本数量: {len(suspicious_n_indices)}")

        if len(suspicious_n_indices) <= 100:
            print(f"可疑样本索引 (前100个): {suspicious_n_indices[:100].tolist()}")


if __name__ == "__main__":
    import sys

    data_dir = "/root/project/ECG/ecg_changgen/train_hospital"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    print(f"检查数据目录: {data_dir}")

    # 加载数据
    print("\n加载数据...")
    X, Y = load_data(data_dir)
    print(f"数据形状: X={X.shape}, Y={Y.shape}")

    # 比较类别特征
    compare_class_features(X, Y)

    # 查找异常样本
    suspicious = find_suspicious_samples(X, Y, n_per_class=50)

    # 检查类别重叠
    check_class_overlap(X, Y)

    print("\n" + "="*80)
    print("建议:")
    print("="*80)
    print("1. 对上述可疑样本进行人工复核")
    print("2. 特别关注那些统计特征与本类差异很大的样本")
    print("3. 检查标注工具的准确性")
    print("4. 考虑引入更多医生进行双盲标注，提高标注质量")
    print("="*80)
