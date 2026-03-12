import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def generate_synthetic_ecg(duration=10, fs=125, heart_rate=75):
    """
    生成一个模拟的心电信号 (仅用于演示)
    Generate a synthetic ECG signal (for demonstration only)
    """
    t = np.linspace(0, duration, duration * fs)
    # 模拟一个简单的周期性信号 (Simulate a simple periodic signal)
    # 使用几个不同频率的正弦波叠加来模拟 P-QRS-T 波形的大致形状
    ecg = 1.0 * np.sin(2 * np.pi * (heart_rate/60) * t)  # 主要心跳周期
    ecg += 0.5 * np.sin(2 * np.pi * (heart_rate/60) * 2 * t + 0.5) # 谐波
    ecg += 0.2 * np.sin(2 * np.pi * (heart_rate/60) * 4 * t + 1.0)
    
    # 添加高斯噪声 (Add Gaussian noise)
    noise = np.random.normal(0, 0.1, len(t))
    
    # 添加基线漂移 (Add baseline wander)
    baseline = 0.5 * np.sin(2 * np.pi * 0.1 * t)
    
    return t, ecg + noise + baseline

def preprocess_ecg(raw_signal, fs=125):
    """
    将长时程 ECG 信号转换为模型所需的单拍切片
    Convert long-duration ECG signal into single-beat segments for the model
    """
    print(f"1. 原始信号长度: {len(raw_signal)} 采样点")
    
    # --- 步骤 1: 去噪 (Denoising) ---
    # 使用带通滤波器 (0.5Hz - 40Hz) 去除基线漂移和高频噪声
    lowcut = 0.5
    highcut = 40.0
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(1, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, raw_signal)
    
    # --- 步骤 2: R波检测 (R-peak Detection) ---
    # 查找峰值，距离至少 0.6秒 (0.6 * fs)
    distance = int(0.6 * fs)
    peaks, _ = signal.find_peaks(filtered_signal, height=0.5, distance=distance)
    print(f"2. 检测到 {len(peaks)} 个心跳 (R波峰值)")
    
    segments = []
    
    # --- 步骤 3: 切片 (Segmentation) ---
    # 目标长度: 187 (与 MIT-BIH 数据集一致)
    target_length = 187
    
    for r_peak in peaks:
        # 截取窗口：R波前 50 点，R波后 137 点 (总共 187)
        # 注意：这个比例 (50/137) 是经验值，取决于 R 波在样本中的期望位置
        start = r_peak - 50
        end = r_peak + 137
        
        # 边界检查
        if start < 0 or end > len(filtered_signal):
            continue
            
        segment = filtered_signal[start:end]
        
        # --- 步骤 4: 归一化 (Normalization) ---
        # Min-Max 归一化到 0-1 之间
        if np.max(segment) != np.min(segment):
            segment = (segment - np.min(segment)) / (np.max(segment) - np.min(segment))
        else:
            segment = np.zeros_like(segment)
            
        segments.append(segment)
        
    print(f"3. 成功提取 {len(segments)} 个有效心拍切片")
    
    return np.array(segments)

if __name__ == "__main__":
    # 1. 生成模拟数据
    fs = 125 # 采样率
    t, raw_ecg = generate_synthetic_ecg(duration=10, fs=fs)
    
    # 2. 运行预处理
    processed_beats = preprocess_ecg(raw_ecg, fs=fs)
    
    # 3. 展示结果形状
    print(f"4. 最终输出形状: {processed_beats.shape}")
    print("   (样本数, 特征长度)")
    
    # 4. 打印第一个心拍的数据示例 (前 10 个点)
    if len(processed_beats) > 0:
        print(f"5. 示例数据 (第一个心拍前10个点):\n   {processed_beats[0][:10]}")
