import pandas as pd
import numpy as np
import argparse
import os
import re
from datetime import timedelta
import sys
from scipy.signal import welch, detrend

def generate_report(record_name, fs=125, patient_info=None, input_dir='input', output_dir='input'):
    print(f"Generating Clinical Report for record: {record_name}...")
    
    # --- 1. Load Data ---
    csv_path = os.path.join(input_dir, f'{record_name}_predictions.csv')
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    r_peaks = df['R_Peak_Pos'].values
    classes = df['Predicted_Class'].values
    
    # Basic Time Calculations
    total_beats = len(r_peaks)
    duration_sec = (r_peaks[-1] - r_peaks[0]) / fs
    duration_str = str(timedelta(seconds=int(duration_sec)))
    
    # RR Intervals (ms)
    rr_intervals = np.diff(r_peaks) / fs * 1000
    # Associated times for RR intervals (time of the second beat in the pair)
    rr_times = r_peaks[1:] / fs # in seconds
    
    # --- 2. Summary Statistics ---
    # Instantaneous Heart Rate (bpm)
    # Filter artifacts for HR calculation to avoid unrealistic extremes
    # Loose filter: 200ms (300bpm) to 3000ms (20bpm).
    # STRICTER FILTER: 30 bpm to 200 bpm for reporting min/max
    # Min RR = 300ms (200 bpm), Max RR = 2000ms (30 bpm)
    
    valid_hr_mask = (rr_intervals >= 300) & (rr_intervals <= 2000)
    valid_rr = rr_intervals[valid_hr_mask]
    valid_rr_times = rr_times[valid_hr_mask]
    
    if len(valid_rr) > 0:
        avg_hr = float(np.mean(60000 / valid_rr))

        rr_series = pd.Series(valid_rr)
        rr_med = rr_series.rolling(window=11, center=True, min_periods=6).median()
        rr_med = rr_med.fillna(float(np.median(valid_rr)))
        hr_clean_mask = (np.abs(rr_series - rr_med) <= (0.2 * rr_med)).values

        hr_rr = valid_rr[hr_clean_mask]
        hr_times = valid_rr_times[hr_clean_mask]

        if len(hr_rr) < 4:
            hr_rr = valid_rr
            hr_times = valid_rr_times

        inst_hr = 60000 / hr_rr

        median_rr_s = float(np.median(hr_rr) / 1000.0) if len(hr_rr) > 0 else 1.0
        window = int(round(10.0 / median_rr_s)) if median_rr_s > 0 else 10
        window = max(4, min(window, 20))

        hr_series = pd.Series(inst_hr)
        smooth_hr_series = hr_series.rolling(window=window, center=True).mean().dropna()
        if len(smooth_hr_series) > 0:
            smooth_hr = smooth_hr_series.values
            smooth_idx = smooth_hr_series.index.values
            smooth_times = hr_times[smooth_idx]

            min_hr = float(np.min(smooth_hr))
            max_hr = float(np.max(smooth_hr))
            min_hr_time = str(timedelta(seconds=int(smooth_times[int(np.argmin(smooth_hr))])))
            max_hr_time = str(timedelta(seconds=int(smooth_times[int(np.argmax(smooth_hr))])))
        else:
            min_hr = float(np.min(inst_hr)) if len(inst_hr) > 0 else 0.0
            max_hr = float(np.max(inst_hr)) if len(inst_hr) > 0 else 0.0
            if len(inst_hr) > 0:
                min_hr_time = str(timedelta(seconds=int(hr_times[int(np.argmin(inst_hr))])))
                max_hr_time = str(timedelta(seconds=int(hr_times[int(np.argmax(inst_hr))])))
            else:
                min_hr_time = max_hr_time = "N/A"
    else:
        min_hr = max_hr = avg_hr = 0
        min_hr_time = max_hr_time = "N/A"

    rr_long_mask = (rr_intervals >= 300) & (rr_intervals <= 3000)
    rr_long = rr_intervals[rr_long_mask]
    rr_long_times = rr_times[rr_long_mask]
    if len(rr_long) > 0:
        rr_long_series = pd.Series(rr_long)
        rr_long_med = rr_long_series.rolling(window=11, center=True, min_periods=6).median()
        rr_long_med = rr_long_med.fillna(float(np.median(rr_long)))
        rr_long_clean_mask = (np.abs(rr_long_series - rr_long_med) <= (0.35 * rr_long_med)).values

        rr_long_clean = rr_long[rr_long_clean_mask]
        rr_long_clean_times = rr_long_times[rr_long_clean_mask]
        if len(rr_long_clean) == 0:
            rr_long_clean = rr_long
            rr_long_clean_times = rr_long_times

        max_rr = float(np.max(rr_long_clean) / 1000.0)
        max_rr_idx = int(np.argmax(rr_long_clean))
        max_rr_time = str(timedelta(seconds=int(rr_long_clean_times[max_rr_idx])))
    else:
        max_rr = 0.0
        max_rr_time = "N/A"

    # --- 3. Arrhythmia Analysis ---
    # Mappings
    # 0: N, 1: S, 2: V, 3: F (Treat as V), 4: Q (Ignore/Other)
    
    # Symbol array for pattern matching
    symbols = np.array(['O'] * len(classes))
    symbols[classes == 0] = 'N'
    symbols[classes == 1] = 'S'
    symbols[classes == 2] = 'V'
    symbols[classes == 3] = 'V' # Treat Fusion as Ventricular for grouping
    symbols[classes == 4] = 'O'
    
    symbol_str = "".join(symbols)
    
    # Helper for counting and MASKING (so we don't double count beats)
    # We will use a boolean mask: True = Available, False = Processed
    beat_available = np.ones(len(classes), dtype=bool)
    
    def analyze_rhythm(target_char, other_char):
        """
        target_char: 'V' or 'S'
        other_char: 'N'
        Returns dict with counts
        """
        stats = {
            'single': 0,
            'pair': 0,
            'bigeminy': 0,
            'trigeminy': 0,
            'tachycardia': 0,
            'total_beats': np.sum((classes == 1) if target_char == 'S' else ((classes == 2) | (classes == 3)))
        }
        
        # 1. Bigeminy (NXNX...) -> Min 3 repeats of NX
        bigeminy_regex = f"(N{target_char}){{3,}}"
        for match in re.finditer(bigeminy_regex, symbol_str):
            stats['bigeminy'] += 1
            start, end = match.span()
            beat_available[start:end] = False

        # 2. Trigeminy (NNXNNX...) -> Min 3 repeats
        trigeminy_regex = f"(NN{target_char}){{3,}}"
        for match in re.finditer(trigeminy_regex, symbol_str):
            start, end = match.span()
            beat_available[start:end] = False
            stats['trigeminy'] += 1

        # 3. Runs (Consecutive X)
        # We look for sequences of X in the *remaining* available beats.
        masked_symbols = []
        for i, char in enumerate(symbols):
            if beat_available[i]:
                masked_symbols.append(char)
            else:
                masked_symbols.append('.') # Placeholder
        
        masked_str = "".join(masked_symbols)
        
        run_regex = f"{target_char}+"
        for match in re.finditer(run_regex, masked_str):
            length = len(match.group(0))
            if length == 1:
                stats['single'] += 1
            elif length == 2:
                stats['pair'] += 1
            elif length >= 3:
                stats['tachycardia'] += 1
                
        return stats

    v_stats = analyze_rhythm('V', 'N')
    s_stats = analyze_rhythm('S', 'N')
    
    # Calculate percentages
    v_percent = (v_stats['total_beats'] / total_beats) * 100
    s_percent = (s_stats['total_beats'] / total_beats) * 100

    # --- 4. HRV Analysis ---
    is_physiological = (rr_intervals >= 300) & (rr_intervals <= 2000)
    rr_phys = rr_intervals[is_physiological]
    rr_phys_times = rr_times[is_physiological]
    rr_phys_idx = np.where(is_physiological)[0]

    if len(rr_phys) > 0:
        rr_series = pd.Series(rr_phys)
        rr_med = rr_series.rolling(window=11, center=True, min_periods=6).median()
        rr_med = rr_med.fillna(float(np.median(rr_phys)))
        nn_rr_mask = np.abs(rr_series - rr_med) <= (0.2 * rr_med)
        clean_nn = rr_phys[nn_rr_mask.values]
        nn_times = rr_phys_times[nn_rr_mask.values]
        nn_idx = rr_phys_idx[nn_rr_mask.values]
    else:
        clean_nn = np.array([])
        nn_times = np.array([])
        nn_idx = np.array([], dtype=int)
    
    # Time Domain
    if len(clean_nn) > 1:
        sdnn = np.std(clean_nn, ddof=1)
        
        # RMSSD & pNN50 (Need adjacent logic)
        consecutive = np.diff(nn_idx) == 1
        all_diffs = np.diff(clean_nn)
        valid_diffs = all_diffs[consecutive]
        
        if len(valid_diffs) > 0:
            rmssd = np.sqrt(np.mean(valid_diffs**2))
            pnn50 = np.sum(np.abs(valid_diffs) > 50) / len(valid_diffs) * 100
        else:
            rmssd = 0
            pnn50 = 0
            
        # SDANN (Standard Deviation of the Averages of NN intervals in all 5-minute segments)
        if len(nn_times) > 0:
            segment_duration = 5 * 60 # 300 seconds
            start_time = nn_times[0]
            end_time = nn_times[-1]
            num_segments = int((end_time - start_time) / segment_duration)
            
            segment_means = []
            for i in range(num_segments + 1):
                seg_start = start_time + i * segment_duration
                seg_end = seg_start + segment_duration
                in_segment = (nn_times >= seg_start) & (nn_times < seg_end)
                segment_nns = clean_nn[in_segment]
                if len(segment_nns) > 30: # Require min 30 beats to count segment
                    segment_means.append(np.mean(segment_nns))
            
            if len(segment_means) > 1:
                sdann = np.std(segment_means, ddof=1)
            else:
                sdann = 0 
        else:
            sdann = 0

        # Triangular Index
        # Integral (Total NN) / Max Density
        if len(clean_nn) > 0:
            bins = np.arange(np.min(clean_nn), np.max(clean_nn) + 8, 8)
            hist, _ = np.histogram(clean_nn, bins=bins)
            max_density = np.max(hist)
            if max_density > 0:
                tri_index = len(clean_nn) / max_density
            else:
                tri_index = 0
        else:
            tri_index = 0
            
    else:
        sdnn = rmssd = pnn50 = sdann = tri_index = 0

    # Frequency Domain (5-min segmented, time-corrected tachogram)
    lf = hf = lfhf = 0
    try:
        if len(clean_nn) > 200 and len(nn_times) == len(clean_nn):
            segment_duration = 5 * 60
            start_time = nn_times[0]
            end_time = nn_times[-1]
            num_segments = int((end_time - start_time) / segment_duration)

            fs_interp = 4.0
            lf_vals = []
            hf_vals = []

            for i in range(num_segments + 1):
                seg_start = start_time + i * segment_duration
                seg_end = seg_start + segment_duration

                in_segment = (nn_times >= seg_start) & (nn_times < seg_end)
                seg_times = nn_times[in_segment]
                seg_nn = clean_nn[in_segment]

                if len(seg_nn) < 60:
                    continue
                if (seg_times[-1] - seg_times[0]) < (segment_duration * 0.8):
                    continue

                t = seg_times - seg_start
                t_interp = np.arange(0, segment_duration, 1 / fs_interp)
                rr_interp = np.interp(t_interp, t, seg_nn)
                rr_interp = detrend(rr_interp, type='linear')

                nperseg = min(256, len(rr_interp))
                if nperseg < 64:
                    continue

                f, pxx = welch(rr_interp, fs=fs_interp, nperseg=nperseg)

                lf_mask = (f >= 0.07) & (f < 0.15)
                hf_mask = (f >= 0.15) & (f <= 0.4)

                lf_i = np.trapz(pxx[lf_mask], f[lf_mask]) if np.any(lf_mask) else 0.0
                hf_i = np.trapz(pxx[hf_mask], f[hf_mask]) if np.any(hf_mask) else 0.0

                if lf_i > 0 and hf_i > 0:
                    lf_vals.append(lf_i)
                    hf_vals.append(hf_i)

            if len(lf_vals) > 0:
                lf = float(np.mean(lf_vals))
                hf = float(np.mean(hf_vals))
                lfhf = (lf / hf) if hf > 0 else 0
            
    except Exception as e:
        print(f"Warning: Manual Frequency analysis failed: {e}")
        lf = hf = lfhf = 0

    # --- 5. Generate Conclusions ---
    conclusions = []
    if v_stats['total_beats'] > 100 or v_stats['tachycardia'] > 0:
        conclusions.append(f"检测到频发室性心律失常 (总占比 {v_percent:.1f}%)，可见 {v_stats['tachycardia']} 次室速。")
    elif v_stats['total_beats'] > 0:
        conclusions.append(f"偶发室性早搏 (总占比 {v_percent:.1f}%)。")
    
    if s_stats['total_beats'] > 100 or s_stats['tachycardia'] > 0:
        conclusions.append(f"检测到频发室上性心律失常 (总占比 {s_percent:.1f}%)。")
        
    if sdnn < 50:
        conclusions.append("HRV 指标 (SDNN) 显著降低，提示自主神经功能受损风险。")
    elif sdnn < 100:
        conclusions.append("HRV 指标 (SDNN) 处于中等水平。")
    else:
        conclusions.append("HRV 指标 (SDNN) 正常，心率变异性良好。")
        
    if len(conclusions) == 0:
        conclusions.append("心律基本正常。")
        
    while len(conclusions) < 2:
        conclusions.append("无其他显著异常发现。")

    # --- 6. Format Output ---
    if patient_info is None:
        patient_info = {
            'name': '未知',
            'sex': '未知',
            'age': '未知'
        }
        
    report_text = f"""
==================================================
              动态心电图检测报告 (模拟)
==================================================
【基本信息】
姓名：{patient_info.get('name', 'N/A')}    性别：{patient_info.get('sex', 'N/A')}    年龄：{patient_info.get('age', 'N/A')}    时长：{duration_str}

【分析统计 - 概要】
总心搏数：{total_beats} 次
最慢心率：{min_hr:.0f} 次/分 (发生于 {min_hr_time})
平均心率：{avg_hr:.0f} 次/分
最快心率：{max_hr:.0f} 次/分 (发生于 {max_hr_time})
最长 RR 间期：{max_rr:.2f} 秒 (发生于 {max_rr_time})

【室性心搏 (V, F, E, I)】
单发：{v_stats['single']} 次
成对：{v_stats['pair']} 对
二联律：{v_stats['bigeminy']} 阵
三联律：{v_stats['trigeminy']} 阵
室性心动过速总数：{v_stats['tachycardia']} 阵
总占比：{v_percent:.2f}%

【室上性心搏 (S, J, A)】
单发：{s_stats['single']} 次
成对：{s_stats['pair']} 对
室上性心动过速总数：{s_stats['tachycardia']} 阵
总占比：{s_percent:.2f}%

【心率变异性 (HRV)】
SDNN：{sdnn:.2f} ms          SDANN：{sdann:.2f} ms
rMSSD：{rmssd:.2f} ms        pNN50：{pnn50:.2f}%
LF/HF：{lfhf:.3f}           三角指数：{tri_index:.2f}
LF：{lf:.2f} ms²             HF：{hf:.2f} ms²

【报告结论】
1. {conclusions[0]}
2. {conclusions[1]}
"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, f"{record_name}_report.txt")
    with open(output_file, "w") as f:
        f.write(report_text)
        
    print(f"Report generated successfully: {output_file}")
    # print(report_text) # Suppress printing to console for batch processing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=str, default='16265', help="Record name or path to .DATA file")
    args = parser.parse_args()
    
    record_arg = args.record
    
    # Check if input is a .DATA file path
    if record_arg.lower().endswith('.data') and os.path.exists(record_arg):
        # Add parent directory to sys.path to import clinical_processor
        # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            import clinical_processor
        except ImportError:
            print("Error: clinical_processor module missing.")
            sys.exit(1)

        print(f"Detected clinical data file: {record_arg}")
        csv_path = clinical_processor.process_clinical_record(record_arg)
        if csv_path:
            basename = os.path.basename(csv_path)
            record_name = basename.replace('_predictions.csv', '')
            print(f"Processed record name: {record_name}")
        else:
            print("Failed to process clinical record.")
            sys.exit(1)
    else:
        # Assume it's a record name (MIT-BIH style) or already processed
        record_name = record_arg

    # Mock patient info
    p_info = {'name': '张三', 'sex': '男', 'age': '58'}
    
    generate_report(record_name, patient_info=p_info)
