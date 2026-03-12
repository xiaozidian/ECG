import wfdb
import neurokit2 as nk
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

def process_and_predict_full_record(record_name='16265'):
    """
    处理完整的 NSRDB 记录，保存为模型可用的格式，并运行预测。
    """
    data_dir = 'input/nsrdb'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # --- 1. 下载数据 (如果不存在) ---
    print(f"[1/8] 检查并下载数据: {record_name}...")
    try:
        wfdb.dl_files('nsrdb', data_dir, [f'{record_name}.hea', f'{record_name}.dat'])
    except Exception as e:
        print(f"下载提示: {e}")

    # --- 2. 读取完整数据 ---
    print(f"[2/8] 读取完整记录...")
    record_path = os.path.join(data_dir, record_name)
    # 读取 header 获取采样率
    header = wfdb.rdheader(record_path)
    fs_original = header.fs
    
    # 读取整个文件
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, 0] # 获取第一通道
    print(f"原始数据长度: {len(signal)} 采样点, 采样率: {fs_original}Hz")
    
    # --- 3. 重采样 (128Hz -> 125Hz) ---
    target_fs = 125
    print(f"[3/8] 重采样至 {target_fs}Hz...")
    # 使用 method='FFT' 对长数据可能较慢，这里 neurokit 默认策略通常够快
    # 如果内存不足，可以考虑分段处理，但对于 ~11M 点，现代机器应该没问题
    signal_resampled = nk.signal_resample(signal, sampling_rate=fs_original, desired_sampling_rate=target_fs)
    print(f"重采样后长度: {len(signal_resampled)} 采样点")
    
    # --- 4. 信号清洗 ---
    print(f"[4/8] 信号清洗 (可能需要几分钟)...")
    # 为了加快速度，我们可以分块处理清洗，但为了保证滤波器状态，最好整段处理
    # 也可以使用简单的带通滤波代替 neurokit 复杂的 pipeline 以加速
    # 这里我们坚持使用 neurokit 以获得最佳质量，但如果太慢可以优化
    ecg_cleaned = nk.ecg_clean(signal_resampled, sampling_rate=target_fs, method="neurokit")
    
    # --- 5. R波检测 ---
    print(f"[5/8] R波检测...")
    _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=target_fs)
    r_peaks_indices = rpeaks['ECG_R_Peaks']
    print(f"检测到 {len(r_peaks_indices)} 个心跳")
    
    # --- 6. 切片与归一化 ---
    print(f"[6/8] 切片与归一化 (Window: -50 ~ +137)...")
    segments = []
    # 预分配数组可能更快，但列表追加比较简单
    for r_peak in r_peaks_indices:
        start = r_peak - 50
        end = r_peak + 137
        if start >= 0 and end < len(ecg_cleaned):
            seg = ecg_cleaned[start:end]
            # 归一化 (0-1)
            min_val = np.min(seg)
            max_val = np.max(seg)
            if max_val - min_val > 0:
                seg = (seg - min_val) / (max_val - min_val)
            else:
                seg = seg - min_val # Flat line case
            segments.append(seg)
            
    X_data = np.array(segments)
    # Reshape for CNN: (N, 187, 1)
    X_data = X_data[..., np.newaxis]
    print(f"最终数据集形状: {X_data.shape}")
    
    # 保存处理好的数据
    output_data_path = f'input/{record_name}_processed.npy'
    np.save(output_data_path, X_data)
    print(f"处理后的数据已保存至: {output_data_path}")
    
    # --- 7. 加载模型与预测 ---
    model_path = 'baseline_cnn_mitbih.h5'
    if os.path.exists(model_path):
        print(f"[7/8] 加载模型 {model_path} 并预测...")
        model = load_model(model_path)
        
        # 批量预测
        predictions = model.predict(X_data, batch_size=1024, verbose=1)
        pred_classes = np.argmax(predictions, axis=1)
        
        # 统计结果
        class_names = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}
        unique, counts = np.unique(pred_classes, return_counts=True)
        print("\n预测结果统计:")
        for cls, count in zip(unique, counts):
            print(f"Class {class_names[cls]} ({cls}): {count} beats")
            
        # --- 8. 保存预测结果 ---
        print(f"[8/8] 保存预测结果...")
        df_res = pd.DataFrame({
            'Beat_Index': range(len(pred_classes)),
            'R_Peak_Pos': r_peaks_indices[:len(pred_classes)], # 对应原始(重采样后)时间点
            'Predicted_Class': pred_classes,
            'Class_Name': [class_names[c] for c in pred_classes]
        })
        output_csv_path = f'input/{record_name}_predictions.csv'
        df_res.to_csv(output_csv_path, index=False)
        print(f"详细预测结果已保存至: {output_csv_path}")
        
    else:
        print(f"警告: 未找到模型文件 {model_path}，跳过预测步骤。")

if __name__ == "__main__":
    process_and_predict_full_record()
