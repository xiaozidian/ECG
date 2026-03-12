"""
快速数据集分析和建议
"""
import numpy as np
import os

def quick_analysis(data_dir):
    print("="*80)
    print("快速数据集分析")
    print("="*80)

    # 只加载标签，速度快
    Y = np.load(os.path.join(data_dir, "Y_0.npy"))

    print(f"\n总样本数: {len(Y):,}")

    unique, counts = np.unique(Y, return_counts=True)
    total = len(Y)

    print("\n类别分布:")
    for cls, count in zip(unique, counts):
        pct = count / total * 100
        print(f"  Class {cls}: {count:>8,} ({pct:>6.2f}%)")

    # 计算如果按8:2划分训练集大小
    print("\n假设训练/测试 = 8:2 划分:")
    for cls, count in zip(unique, counts):
        train_count = int(count * 0.8)
        test_count = count - train_count
        print(f"  Class {cls}: 训练={train_count:>6,}, 测试={test_count:>6,}")

    # 分析问题
    print("\n" + "="*80)
    print("问题诊断")
    print("="*80)

    min_count = counts.min()
    max_count = counts.max()
    imbalance_ratio = max_count / min_count

    print(f"\n1. 类别不平衡比例: {imbalance_ratio:.1f}:1")

    if imbalance_ratio > 100:
        print("   ⚠️ 严重不平衡！")
        print("   - 这个不平衡程度会导致模型严重偏向多数类")
        print("   - 即使使用重采样，少数类样本太少也无法学到有效特征")

    print(f"\n2. 最少类别样本数: {min_count:,}")

    if min_count < 5000:
        print("   ⚠️ 样本数严重不足！")
        print("   - 深度学习通常需要每类至少5,000-10,000个样本")
        print("   - 当前少数类样本数量不足以训练可靠的模型")

    # 分析当前采样策略
    print("\n" + "="*80)
    print("当前采样策略分析")
    print("="*80)

    print("\n您使用的配置: RESAMPLE_WEIGHTS=0.85,0.10,0.05")
    print(f"batch_size=4096 时，期望每个batch的类别分布:")
    print(f"  Class 0: {int(4096 * 0.85):>4} 样本")
    print(f"  Class 1: {int(4096 * 0.10):>4} 样本")
    print(f"  Class 2: {int(4096 * 0.05):>4} 样本")

    print("\n实际数据分布:")
    for cls, count in zip(unique, counts):
        pct = count / total
        print(f"  Class {cls}: {pct:.4f} ({pct*100:.2f}%)")

    print("\n⚠️ 问题所在:")
    print("  - batch中S和V类占比(10%+5%=15%)远高于实际分布(0.5%)")
    print("  - 模型学习到的是采样后的分布，不是真实分布")
    print("  - 导致模型在推理时过度预测S和V类")
    print("  - 表现为: recall尚可，但precision极低")

    # 给出建议
    print("\n" + "="*80)
    print("解决方案建议")
    print("="*80)

    print("\n【优先级1 - 数据层面】必须解决:")
    print("  1. 收集更多S和V类样本")
    print("     - 目标: 每类至少10,000个样本")
    print("     - 可以从其他医院或数据源获取")
    print("     - 或者重新审查标注，找出被误标为N的S和V样本")

    print("\n  2. 人工复核标注质量")
    print("     - 随机抽查100-200个样本，验证标注准确性")
    print("     - 特别关注S和V类样本的标注")
    print("     - 检查是否有过于保守的标注（把S/V标成N）")

    print("\n  3. 考虑使用公开数据集")
    print("     - MIT-BIH Arrhythmia Database")
    print("     - AAMI EC57 标准")
    print("     - 先用公开数据集验证模型架构")

    print("\n【优先级2 - 训练策略】在收集更多数据前的临时措施:")
    print("  1. 降低重采样强度")
    print("     --resample-weights 0.96,0.02,0.02")
    print("     （更接近真实分布）")

    print("\n  2. 使用更强的focal loss")
    print("     --loss focal --focal-gamma 3.0 --focal-alpha auto")

    print("\n  3. 提高决策阈值")
    print("     --s-threshold 0.80 --v-threshold 0.85")

    print("\n  4. 使用更小的模型避免过拟合")
    print("     （当前数据量不足以训练深度模型）")

    print("\n  5. 尝试传统机器学习方法")
    print("     - Random Forest / XGBoost")
    print("     - 基于手工特征（心率变异性、QRS波形等）")
    print("     - 小样本情况下可能比深度学习效果更好")

    print("\n【优先级3 - 评估指标】")
    print("  当前情况下，应该关注:")
    print("  - 使用F1-score作为主要指标（不是accuracy）")
    print("  - 每个类别单独评估")
    print("  - 关注precision-recall曲线，找到最优工作点")

    # 预测建议配置的效果
    print("\n" + "="*80)
    print("预期效果分析")
    print("="*80)

    print("\n⚠️ 现实的期望:")
    print("  在当前数据集上（867个S，1085个V），即使采用最优策略：")
    print("  - S类 F1-score 可能只能达到 0.10-0.20")
    print("  - V类 F1-score 可能只能达到 0.20-0.30")
    print("  - 这是由于样本数量根本性不足导致的")

    print("\n✓ 要达到可用效果 (F1 > 0.70)，需要:")
    print("  - 每类至少 10,000+ 样本")
    print("  - 高质量的标注")
    print("  - 多样化的数据来源")

if __name__ == "__main__":
    import sys
    data_dir = "/root/project/ECG/ecg_changgen/train_hospital"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    quick_analysis(data_dir)
