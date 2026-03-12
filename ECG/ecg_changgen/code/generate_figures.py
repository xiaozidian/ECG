"""
生成可视化图表，用于报告
"""
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_comparison_charts(json_path, output_dir):
    """创建对比图表"""

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # 1. 类别分布饼图
    fig, ax = plt.subplots(figsize=(10, 8))

    classes = []
    sizes = []
    colors = ['#2ecc71', '#e74c3c', '#f39c12']

    for cls_id in sorted(data['class_distribution'].keys()):
        cls_info = data['class_distribution'][str(cls_id)]
        classes.append(cls_info['name'])
        sizes.append(cls_info['total'])

    # 只显示小类别的百分比标签
    def autopct_format(pct):
        if pct < 1:
            return f'{pct:.2f}%'
        else:
            return f'{pct:.1f}%'

    wedges, texts, autotexts = ax.pie(sizes, labels=classes, colors=colors,
                                        autopct=autopct_format, startangle=90)

    # 使小类别的标签更清晰
    for i, (text, autotext) in enumerate(zip(texts, autotexts)):
        if sizes[i] / sum(sizes) < 0.01:
            text.set_fontsize(10)
            text.set_weight('bold')
            autotext.set_fontsize(9)
            autotext.set_weight('bold')
            autotext.set_color('white')

    ax.set_title('Current Dataset Class Distribution\n(Severe Imbalance: 460:1)', fontsize=14, weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_class_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"Generated: 1_class_distribution.png")
    plt.close()

    # 2. 训练样本数对比图
    fig, ax = plt.subplots(figsize=(12, 6))

    class_names = []
    train_counts = []
    test_counts = []

    for cls_id in sorted(data['class_distribution'].keys()):
        cls_info = data['class_distribution'][str(cls_id)]
        class_names.append(cls_info['name'])
        train_counts.append(cls_info['train'])
        test_counts.append(cls_info['test'])

    x = np.arange(len(class_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, train_counts, width, label='Training Set', color='#3498db')
    bars2 = ax.bar(x + width/2, test_counts, width, label='Test Set', color='#95a5a6')

    # 添加最低要求线
    ax.axhline(y=5000, color='r', linestyle='--', linewidth=2, label='Minimum Requirement (5,000)')

    ax.set_xlabel('Class', fontsize=12, weight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, weight='bold')
    ax.set_title('Sample Count per Class vs. Minimum Requirement', fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()

    # 在柱子上显示数值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_sample_counts.png'), dpi=300, bbox_inches='tight')
    print(f"Generated: 2_sample_counts.png")
    plt.close()

    # 3. 与MIT-BIH对比
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['S Class', 'V Class']
    mitbih = [data['mitbih_comparison']['S']['mitbih'],
              data['mitbih_comparison']['V']['mitbih']]
    current = [data['mitbih_comparison']['S']['current'],
               data['mitbih_comparison']['V']['current']]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, mitbih, width, label='MIT-BIH (Standard)', color='#27ae60')
    bars2 = ax.bar(x + width/2, current, width, label='Current Dataset', color='#e74c3c')

    ax.set_xlabel('Class', fontsize=12, weight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, weight='bold')
    ax.set_title('Comparison with MIT-BIH Arrhythmia Database', fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # 显示数值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', fontsize=10)

    # 添加百分比标注
    for i, cat in enumerate(categories):
        pct = (current[i] / mitbih[i] * 100)
        ax.text(i, max(mitbih[i], current[i]) * 1.1,
               f'{pct:.1f}%',
               ha='center', fontsize=11, weight='bold', color='red')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_mitbih_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Generated: 3_mitbih_comparison.png")
    plt.close()

    # 4. 训练样本 vs 要求的热力图
    fig, ax = plt.subplots(figsize=(10, 6))

    requirements = [
        ("Minimum (Time Series)", 5000),
        ("Recommended (Time Series)", 10000),
        ("Medical Data Standard", 50000)
    ]

    class_labels = []
    for cls_id in sorted(data['class_distribution'].keys()):
        if cls_id != 0:  # 只看S和V类
            cls_info = data['class_distribution'][str(cls_id)]
            class_labels.append(cls_info['name'].split()[0])  # 只取简称

    # 创建数据矩阵
    matrix = []
    for req_name, req_value in requirements:
        row = []
        for cls_id in sorted(data['class_distribution'].keys()):
            if cls_id != 0:
                cls_info = data['class_distribution'][str(cls_id)]
                train_count = cls_info['train']
                ratio = min(train_count / req_value, 1.0)  # 最大为1
                row.append(ratio)
        matrix.append(row)

    matrix = np.array(matrix)

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # 设置刻度
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(requirements)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels([req[0] for req in requirements])

    # 在每个单元格中显示百分比
    for i in range(len(requirements)):
        for j in range(len(class_labels)):
            text = ax.text(j, i, f'{matrix[i, j]*100:.1f}%',
                          ha="center", va="center", color="black" if matrix[i, j] > 0.5 else "white",
                          fontsize=12, weight='bold')

    ax.set_title('Sample Adequacy Heatmap\n(% of Requirement Met)', fontsize=14, weight='bold')
    fig.colorbar(im, ax=ax, label='Adequacy Ratio')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_adequacy_heatmap.png'), dpi=300, bbox_inches='tight')
    print(f"Generated: 4_adequacy_heatmap.png")
    plt.close()

    # 5. 性能对比图（模拟的）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # F1-score对比
    classes = ['S Class', 'V Class']
    current_f1 = [0.0466, 0.1567]  # 从实验结果
    expected_f1 = [0.70, 0.70]  # 期望值

    x = np.arange(len(classes))
    width = 0.35

    ax1.bar(x - width/2, current_f1, width, label='Current Result', color='#e74c3c')
    ax1.bar(x + width/2, expected_f1, width, label='Expected (w/ adequate data)', color='#27ae60')

    ax1.set_ylabel('F1-Score', fontsize=12, weight='bold')
    ax1.set_title('Model Performance: Current vs Expected', fontsize=14, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.legend()
    ax1.set_ylim([0, 1.0])

    # 添加数值标签
    for i, (curr, exp) in enumerate(zip(current_f1, expected_f1)):
        ax1.text(i - width/2, curr + 0.02, f'{curr:.2f}', ha='center', fontsize=10)
        ax1.text(i + width/2, exp + 0.02, f'{exp:.2f}', ha='center', fontsize=10)

    # Precision对比
    current_precision = [0.0245, 0.0872]
    expected_precision = [0.70, 0.70]

    ax2.bar(x - width/2, current_precision, width, label='Current Result', color='#e74c3c')
    ax2.bar(x + width/2, expected_precision, width, label='Expected (w/ adequate data)', color='#27ae60')

    ax2.set_ylabel('Precision', fontsize=12, weight='bold')
    ax2.set_title('Precision: Current vs Expected', fontsize=14, weight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)
    ax2.legend()
    ax2.set_ylim([0, 1.0])

    # 添加数值标签和警告
    for i, (curr, exp) in enumerate(zip(current_precision, expected_precision)):
        ax2.text(i - width/2, curr + 0.02, f'{curr:.2f}', ha='center', fontsize=10)
        ax2.text(i + width/2, exp + 0.02, f'{exp:.2f}', ha='center', fontsize=10)

    # 添加不可用区域标注
    ax2.axhspan(0, 0.5, alpha=0.2, color='red', label='Clinically Unusable')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_performance_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Generated: 5_performance_comparison.png")
    plt.close()

    print("\n所有图表生成完成！")


if __name__ == "__main__":
    import sys

    json_path = "/root/project/ECG/ecg_changgen/dataset_summary.json"
    output_dir = "/root/project/ECG/ecg_changgen/report_figures"

    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    print("="*80)
    print("生成报告图表")
    print("="*80)

    create_comparison_charts(json_path, output_dir)

    print("\n" + "="*80)
    print(f"图表已保存到: {output_dir}")
    print("="*80)
