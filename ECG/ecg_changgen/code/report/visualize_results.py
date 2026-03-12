import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import neurokit2 as nk
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def perform_clustering(beats, n_clusters=3, use_pca=True, outlier_corr_threshold=0.65):
    """
    对波形进行形态学聚类，包含特征降维和相关性校验
    """
    if len(beats) < n_clusters * 2:
        return np.zeros(len(beats), dtype=int), [np.mean(beats, axis=0)]
    
    # --- Step 1: 特征提取 (PCA) ---
    if use_pca:
        # 保留足够的主成分 (例如 15 或 95% 能量)
        n_components = min(15, beats.shape[1], len(beats)-1)
        pca = PCA(n_components=n_components, random_state=42)
        beats_features = pca.fit_transform(beats)
    else:
        beats_features = beats
    
    # --- Step 2: KMeans 聚类 ---
    # 如果数据量过大，先采样进行聚类训练
    if len(beats) > 10000:
        train_indices = np.random.choice(len(beats), 10000, replace=False)
        train_features = beats_features[train_indices]
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        kmeans.fit(train_features)
        labels = kmeans.predict(beats_features)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(beats_features)
        
    # --- Step 3: 模版相关性校验 (Refinement) ---
    # 计算初始质心 (原始空间)
    centroids = []
    for k in range(n_clusters):
        cluster_beats = beats[labels == k]
        if len(cluster_beats) > 0:
            centroids.append(np.mean(cluster_beats, axis=0))
        else:
            centroids.append(np.zeros(beats.shape[1]))
    
    new_labels = labels.copy()
    
    for k in range(n_clusters):
        idx = np.where(labels == k)[0]
        if len(idx) == 0: continue
        
        cluster_data = beats[idx]
        centroid = centroids[k]
        
        # 计算 Pearson 相关系数 (Vectorized)
        # Center data
        cluster_centered = cluster_data - cluster_data.mean(axis=1, keepdims=True)
        centroid_centered = centroid - centroid.mean()
        
        # Norms
        cluster_norms = np.linalg.norm(cluster_centered, axis=1)
        centroid_norm = np.linalg.norm(centroid_centered)
        
        # 防止除零
        valid_mask = (cluster_norms > 1e-6) & (centroid_norm > 1e-6)
        corrs = np.zeros(len(idx))
        
        if centroid_norm > 1e-6:
             dots = np.dot(cluster_centered, centroid_centered)
             # 只计算 valid 的
             corrs[valid_mask] = dots[valid_mask] / (cluster_norms[valid_mask] * centroid_norm)
        
        # 应用阈值
        outlier_mask = corrs < outlier_corr_threshold
        new_labels[idx[outlier_mask]] = -1
        
        # 0.8 < rho < 0.95 -> 变异类 (保留在类中，但可以在后续统计中体现)
        # 这里我们暂不修改标签，只剔除离群点
        
    # --- Step 4: 基于清洗后的数据重新计算质心 ---
    final_centroids = []
    for k in range(n_clusters):
        # 只用非离群点计算质心
        valid_idx = (new_labels == k)
        if np.sum(valid_idx) > 0:
            final_centroids.append(np.mean(beats[valid_idx], axis=0))
        else:
            final_centroids.append(centroids[k]) # Fallback
            
    return new_labels, np.array(final_centroids)

def plot_single_class_detail(record_name, class_name, beats, labels, centroids, base_color):
    """
    为单个类别生成详细的形态学分析图
    """
    n_clusters = len(centroids)
    
    # 创建画布: 上半部分为所有聚类中心的对比，下半部分为各聚类的详细展示
    fig = plt.figure(figsize=(15, 10))
    # 布局: 2行，n_clusters列
    # 如果 n_clusters 只有 1，则布局简单点
    if n_clusters == 1:
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        
        # 背景
        sample_indices = np.random.choice(len(beats), min(200, len(beats)), replace=False)
        for si in sample_indices:
            ax.plot(beats[si], color='gray', alpha=0.1, linewidth=0.5)
            
        # 平均线
        ax.plot(centroids[0], color=base_color, linewidth=2.5, label='Average')
        ax.set_title(f"Class {class_name} (n={len(beats)})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    else:
        gs = fig.add_gridspec(2, n_clusters)
        
        # --- Top: Comparison (Span all columns) ---
        ax_all = fig.add_subplot(gs[0, :])
        linestyles = ['-', '--', '-.', ':']
        
        # 统计各类别数量
        counts = []
        for c_idx in range(len(centroids)):
            counts.append(np.sum(labels == c_idx))
        
        # 统计噪声
        noise_count = np.sum(labels == -1)
        
        valid_total = int(np.sum(np.array(counts)))
        if valid_total > 0:
            percentages = [c / valid_total * 100 for c in counts]
        else:
            percentages = [0.0 for _ in counts]

        percentages_rounded = [round(p, 1) for p in percentages]
        current_sum = sum(percentages_rounded)
        if abs(current_sum - 100.0) > 0.01 and valid_total > 0:
            diff = 100.0 - current_sum
            max_idx = int(np.argmax(percentages_rounded))
            percentages_rounded[max_idx] += diff

        noise_percentage_total_rounded = round(noise_count / len(beats) * 100, 1) if len(beats) > 0 else 0.0
        
        title_suffix = f" (Outliers/Noise: {noise_count})" if noise_count > 0 else ""
        
        for c_idx, centroid in enumerate(centroids):
            ls = linestyles[c_idx % len(linestyles)]
            # 使用修正后的百分比
            pct = percentages_rounded[c_idx]
            ax_all.plot(centroid, linestyle=ls, linewidth=2, label=f'Type {c_idx+1} (non-outlier: {pct:.1f}%)')
            
        # 添加离群点到图例
        if noise_count > 0:
            ax_all.plot([], [], ' ', label=f'Outliers (of total: {noise_percentage_total_rounded:.1f}%)')
        
        ax_all.set_title(f"Class {class_name} - Morphologies Comparison{title_suffix}")
        ax_all.legend()
        ax_all.grid(True, alpha=0.3)
        
        # --- Bottom: Individual Clusters ---
        for c_idx, centroid in enumerate(centroids):
            ax = fig.add_subplot(gs[1, c_idx])
            
            # Background samples for this cluster
            cluster_indices = np.where(labels == c_idx)[0]
            if len(cluster_indices) > 0:
                sample_indices = np.random.choice(cluster_indices, min(100, len(cluster_indices)), replace=False)
                for si in sample_indices:
                    ax.plot(beats[si], color='gray', alpha=0.05, linewidth=0.5)
            
            # Centroid
            ls = linestyles[c_idx % len(linestyles)]
            ax.plot(centroid, color=base_color, linestyle=ls, linewidth=2.5)
            
            # 使用修正后的百分比
            pct = percentages_rounded[c_idx]
            count = np.sum(labels == c_idx)
            ax.set_title(f"Type {c_idx+1}\n{pct:.1f}% of non-outlier (n={count})")
            ax.grid(True, alpha=0.3)
            if c_idx > 0:
                ax.set_yticks([])
                
    plt.tight_layout()
    # Save
    safe_name = class_name.split(' ')[0] # 'N', 'S' etc
    out_path = f'input/{record_name}_detail_{safe_name}.png'
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存详细分类图: {out_path}")

def visualize_ecg_analysis(record_name='16265', start_sec=100, duration_sec=10, n_subclusters=3, outlier_corr_threshold=0.65):
    """
    可视化 ECG 分析结果，模仿专业软件界面。
    包含：
    1. 各类心拍的平均波形 (Templates)
    2. 各类心拍的叠加波形 (Superimposed)
    3. 带有分类标注的心律条图 (Rhythm Strip)
    
    参数:
    - n_subclusters: 每个类别形态学聚类的簇数 (默认为 3)
    """
    
    # 1. 加载数据
    print("正在加载数据...")
    npy_path = f'input/{record_name}_processed.npy'
    csv_path = f'input/{record_name}_predictions.csv'
    
    if not os.path.exists(npy_path) or not os.path.exists(csv_path):
        print("错误: 找不到处理后的数据或预测结果，请先运行 process_full_nsrdb.py")
        return

    # 加载波形数据 (N, 187, 1) -> (N, 187)
    beats_data = np.load(npy_path).squeeze()
    # 加载预测结果
    df_pred = pd.read_csv(csv_path)
    
    # 类别映射
    class_map = {0: 'N (Normal)', 1: 'S (Supraventricular)', 2: 'V (Ventricular)', 3: 'F (Fusion)', 4: 'Q (Unknown)'}
    color_map = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:red', 3: 'tab:green', 4: 'tab:purple'}
    
    # 创建画布
    fig = plt.figure(figsize=(18, 10))
    # 使用 GridSpec 布局
    # Top row: Templates & Superimposed (占用 1/3 高度)
    # Bottom row: Rhythm Strip (占用 2/3 高度)
    gs = fig.add_gridspec(3, 5)
    
    print("正在绘制模板和叠加波形...")
    
    # --- Part 1: 各类别的平均波形 (Templates) & 形态学聚类 ---
    unique_classes = sorted(df_pred['Predicted_Class'].unique())
    
    for idx, cls_id in enumerate(unique_classes):
        if idx >= 5: break # 最多显示5列
        
        # 获取该类的所有波形索引
        indices = df_pred[df_pred['Predicted_Class'] == cls_id].index
        cls_beats = beats_data[indices]
        
        ax = fig.add_subplot(gs[0, idx])
        
        # 准备变量用于单独绘图
        labels = np.zeros(len(cls_beats), dtype=int)
        centroids = [np.mean(cls_beats, axis=0)]
        
        # 对 N 类 (0) 或样本数足够的类进行聚类分析
        if (cls_id == 0 or len(cls_beats) > 100) and n_subclusters > 1:
            print(f"对类别 {class_map[cls_id]} 进行形态学聚类 (k={n_subclusters})...")
            labels, centroids = perform_clustering(
                cls_beats,
                n_clusters=n_subclusters,
                outlier_corr_threshold=outlier_corr_threshold,
            )
            
            # 使用渐变色或不同线型区分聚类
            base_color = color_map[cls_id]
            
            counts = [int(np.sum(labels == c_idx)) for c_idx in range(len(centroids))]
            valid_total = int(np.sum(np.array(counts)))
            if valid_total > 0:
                percentages = [c / valid_total * 100 for c in counts]
            else:
                percentages = [0.0 for _ in counts]

            percentages_rounded = [round(p, 1) for p in percentages]
            current_sum = sum(percentages_rounded)
            if abs(current_sum - 100.0) > 0.01 and valid_total > 0:
                diff = 100.0 - current_sum
                max_idx = int(np.argmax(percentages_rounded))
                percentages_rounded[max_idx] += diff

            for c_idx, centroid in enumerate(centroids):
                pct = percentages_rounded[c_idx]
                count = counts[c_idx]

                # 绘制聚类中心
                linestyles = ['-', '--', '-.', ':']
                ls = linestyles[c_idx % len(linestyles)]
                
                ax.plot(centroid, color=base_color, linestyle=ls, linewidth=2, 
                        label=f'Type {c_idx+1} (non-outlier: {pct:.1f}%)')
                
                # 绘制该聚类的一些背景样本
                cluster_indices = np.where(labels == c_idx)[0]
                sample_indices = np.random.choice(cluster_indices, min(20, len(cluster_indices)), replace=False)
                for si in sample_indices:
                    ax.plot(cls_beats[si], color='gray', alpha=0.05, linewidth=0.5)
            
            # 统计离群点并添加到图例
            noise_count = np.sum(labels == -1)
            if noise_count > 0:
                noise_percentage = noise_count / len(cls_beats) * 100
                ax.plot([], [], ' ', label=f'Outliers (of total: {noise_percentage:.1f}%)')
            
            ax.set_title(f"{class_map[cls_id]}\nTotal: {len(indices)} (Clustered)")
            ax.legend(fontsize='x-small')
            
        else:
            # 样本太少，仅画平均波形
            mean_beat = centroids[0]
            
            # 叠加背景
            sample_size = min(50, len(cls_beats))
            random_indices = np.random.choice(len(cls_beats), sample_size, replace=False)
            for i in random_indices:
                ax.plot(cls_beats[i], color='gray', alpha=0.1, linewidth=0.5)
                
            # 画平均线
            ax.plot(mean_beat, color=color_map[cls_id], linewidth=2, label='Avg')
            ax.set_title(f"{class_map[cls_id]}\nCount: {len(indices)}")
        
        # 调用单独绘图函数
        plot_single_class_detail(record_name, class_map[cls_id], cls_beats, labels, centroids, color_map[cls_id])
        
        ax.set_xticks([])
        if idx > 0: ax.set_yticks([])
        ax.grid(True, alpha=0.3)

    # --- Part 2: 带有标注的心律条图 (Rhythm Strip) ---
    print("正在绘制心律条图...")
    ax_strip = fig.add_subplot(gs[1:, :]) # 占用下方所有列
    
    # 读取原始信号的一段用于展示
    data_dir = 'input/nsrdb'
    record_path = os.path.join(data_dir, record_name)
    
    # 计算读取范围
    header = wfdb.rdheader(record_path)
    fs = header.fs
    samp_from = int(start_sec * fs)
    samp_to = int((start_sec + duration_sec) * fs)
    
    record = wfdb.rdrecord(record_path, sampfrom=samp_from, sampto=samp_to)
    signal = record.p_signal[:, 0]
    time_axis = np.arange(samp_from, samp_to) / fs
    
    # 绘制原始信号
    ax_strip.plot(time_axis, signal, color='black', linewidth=0.8, label='Raw Signal')
    
    # 找到该时间段内的 R 波和预测结果
    # R_Peak_Pos 是相对于原始信号的索引 (注意：如果是重采样后的索引需要转换，但之前代码保存的是 indices，需确认)
    # 回顾 process_full_nsrdb.py: 
    # r_peaks_indices 是基于 heavy cleaned & resampled (125Hz) 的信号检测的。
    # 而这里我们读取的是原始 (128Hz) 信号。
    # 这是一个关键点！为了准确对应，我们需要把 CSV 里的 R_Peak_Pos (125Hz域) 映射回 128Hz 域，或者直接展示重采样后的信号。
    # 为了简化且准确，我们这里展示 "重采样并清洗后" 的信号片段，这样能完美对齐。
    
    # --- 修正: 重新生成一段重采样后的信号用于展示，确保对齐 ---
    # 为了展示效果，我们重新读取原始数据并重采样这一小段，但这可能导致边缘效应。
    # 更好的方法是：假设 R_Peak_Pos 是在 125Hz 下的。
    # 我们画图的 X 轴以 125Hz 的采样点为单位即可。
    
    # 模拟获取对应的信号片段 (为了准确，我们直接从 processed beats 拼凑有点难，还是读原始数据并重采样最稳)
    # 但重采样整个文件太慢。
    # 策略：直接读取原始文件，然后根据 fs_ratio 缩放 R_Peak_Pos 坐标来标注。
    
    fs_model = 125
    fs_original = 128
    ratio = fs_original / fs_model
    
    # 筛选在当前显示窗口内的 R 波
    # df_pred['R_Peak_Pos'] 是 125Hz 下的索引
    # 转换到原始 128Hz 下的索引
    df_pred['R_Peak_Pos_Original'] = df_pred['R_Peak_Pos'] * ratio
    
    mask = (df_pred['R_Peak_Pos_Original'] >= samp_from) & (df_pred['R_Peak_Pos_Original'] < samp_to)
    subset_pred = df_pred[mask]
    
    # 在条图上标注
    for _, row in subset_pred.iterrows():
        pos = row['R_Peak_Pos_Original']
        cls = int(row['Predicted_Class'])
        time_sec = pos / fs_original
        
        # 获取该时刻的信号幅值 (近似)
        idx_in_signal = int(pos - samp_from)
        if 0 <= idx_in_signal < len(signal):
            val = signal[idx_in_signal]
            
            # 画竖线标记
            ax_strip.axvline(x=time_sec, color=color_map[cls], linestyle='--', alpha=0.5, ymin=0.2, ymax=0.8)
            # 画文字标签
            ax_strip.text(time_sec, val + 0.1, class_map[cls].split(' ')[0], 
                          color=color_map[cls], fontsize=12, fontweight='bold', ha='center')
            
            # 高亮异常心跳背景
            if cls != 0: # 非正常心跳
                ax_strip.axvspan(time_sec - 0.2, time_sec + 0.2, color=color_map[cls], alpha=0.2)

    ax_strip.set_title(f"Rhythm Strip Analysis ({start_sec}s - {start_sec+duration_sec}s)", fontsize=14)
    ax_strip.set_xlabel("Time (seconds)")
    ax_strip.set_ylabel("Amplitude (mV)")
    ax_strip.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_strip.minorticks_on()
    
    plt.tight_layout()
    output_file = f'input/{record_name}_visualization_report.png'
    plt.savefig(output_file, dpi=150)
    print(f"可视化报告已保存至: {output_file}")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ECG Visualization with Morphological Clustering')
    parser.add_argument('--record', type=str, default='16265', help='Record name (e.g., 16265)')
    parser.add_argument('--start', type=float, default=None, help='Start time in seconds')
    parser.add_argument('--duration', type=float, default=10, help='Duration in seconds')
    parser.add_argument('--clusters', type=int, default=3, help='Number of sub-clusters for morphology (default: 3)')
    parser.add_argument('--outlier_corr', type=float, default=0.7, help='Correlation threshold for outliers (default: 0.7)')
    
    args = parser.parse_args()
    
    record_name = args.record
    n_clusters = args.clusters
    
    # 找一个包含异常心跳的时间段展示 (例如室性早搏 V)
    # 先读取 CSV 看看哪里有 V 类 (Class 2)
    csv_path = f'output/{record_name}_predictions.csv'
    
    start_time = args.start
    
    if start_time is None:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            v_beats = df[df['Predicted_Class'] == 2]
            if not v_beats.empty:
                # 取第一个 V beat 的位置
                first_v_idx = v_beats.iloc[0]['R_Peak_Pos']
                # 转换为秒 (125Hz)
                start_time = max(0, (first_v_idx / 125) - 3) # 从该心跳前3秒开始
                print(f"检测到 V 类心跳，自动定位到 {start_time:.1f}s")
            else:
                print("未检测到 V 类心跳，展示默认开头片段。")
                start_time = 100
        else:
            start_time = 100
            
    visualize_ecg_analysis(
        record_name=record_name,
        start_sec=start_time,
        duration_sec=args.duration,
        n_subclusters=n_clusters,
        outlier_corr_threshold=args.outlier_corr,
    )
