import json
import os
import argparse
import pandas as pd
import numpy as np

def generate_markdown_report(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    md_lines = []
    md_lines.append("# ECG 算法验证报告")
    md_lines.append(f"**日期:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    md_lines.append(f"**总记录数:** {len(data)}")
    md_lines.append("")

    # --- 1. Executive Summary (MAE) ---
    md_lines.append("## 1. 执行摘要 (整体性能)")
    md_lines.append("下表展示了所有验证记录的平均绝对误差 (MAE) 和平均百分比误差 (MPE)，以及**最大单例误差 (Max Single Error)**，用于评估最坏情况下的表现。")
    md_lines.append("")
    
    # Collect all metrics
    if not data:
        print("No data found.")
        return

    first_record_diffs = data[0]['diffs']
    metrics = list(first_record_diffs.keys())
    
    summary_stats = []
    
    # Translation map for metrics
    metric_map = {
        'total_beats': '总心搏数',
        'avg_hr': '平均心率',
        'max_hr': '最大心率',
        'min_hr': '最小心率',
        'max_rr': '最大 RR 间期',
        'ventricular_total': '室性总数',
        'ventricular_single': '室性单发',
        'ventricular_pair': '室性成对',
        'ventricular_bigeminy': '室性二联律',
        'ventricular_trigeminy': '室性三联律',
        'ventricular_vt': '室速 (VT)',
        'supraventricular_total': '房性总数',
        'supraventricular_single': '房性单发',
        'supraventricular_pair': '房性成对',
        'supraventricular_bigeminy': '房性二联律',
        'supraventricular_trigeminy': '房性三联律',
        'supraventricular_svt': '房速 (SVT)',
        'sdnn': 'SDNN',
        'sdann': 'SDANN',
        'rmssd': 'RMSSD',
        'pnn50': 'pNN50',
        'lf': 'LF',
        'hf': 'HF',
        'lf_hf': 'LF/HF',
        'tri_index': '三角指数'
    }

    for metric in metrics:
        abs_diffs = []
        pct_diffs = []
        max_single_err = 0
        max_single_err_id = ""

        for record in data:
            m_data = record['diffs'].get(metric)
            if m_data:
                curr_abs = abs(m_data['abs_diff'])
                abs_diffs.append(curr_abs)
                
                if curr_abs > max_single_err:
                    max_single_err = curr_abs
                    max_single_err_id = record['record_id']

                # Handle cases where pct_diff might be infinite or huge due to div by zero
                if abs(m_data['pct_diff']) < 10000: # Filter out extreme outliers for summary
                    pct_diffs.append(m_data['pct_diff'])
        
        mae = np.mean(abs_diffs) if abs_diffs else 0
        mpe = np.mean(pct_diffs) if pct_diffs else 0
        
        summary_stats.append({
            "Metric": metric,
            "MAE": mae,
            "Mean % Error": mpe,
            "Max Single Err": max_single_err,
            "Worst Case ID": max_single_err_id
        })
    
    df_summary = pd.DataFrame(summary_stats)
    
    # Format the table
    md_lines.append("| 指标 | MAE (平均绝对误差) | 平均百分比误差 | 最大单例误差 (Worst Case) |")
    md_lines.append("| :--- | :--- | :--- | :--- |")
    
    for _, row in df_summary.iterrows():
        metric_key = row['Metric']
        metric_name = metric_map.get(metric_key, metric_key)
        mae_val = f"{row['MAE']:.2f}"
        mpe_val = f"{row['Mean % Error']:.2f}%"
        worst_val = f"{row['Max Single Err']:.2f} (ID: {row['Worst Case ID']})"
        md_lines.append(f"| {metric_name} | {mae_val} | {mpe_val} | {worst_val} |")
        
    md_lines.append("")

    # --- 2. Detailed Record Analysis ---
    md_lines.append("## 2. 详细记录分析")
    
    for record in data:
        rec_id = record['record_id']
        md_lines.append(f"### 记录编号: {rec_id}")
        
        md_lines.append("| 指标 | 基准值 (GT) | 计算值 (Calc) | 绝对误差 | 百分比误差 |")
        md_lines.append("| :--- | :--- | :--- | :--- | :--- |")
        
        diffs = record['diffs']
        for metric in metrics:
            m_data = diffs.get(metric)
            if not m_data: continue
            
            name = metric_map.get(metric, metric)
            gt = m_data['gt']
            calc = m_data['calc']
            abs_d = m_data['abs_diff']
            pct_d = m_data['pct_diff']
            
            # Formatting
            if isinstance(gt, float): gt_str = f"{gt:.2f}"
            else: gt_str = str(gt)
            
            if isinstance(calc, float): calc_str = f"{calc:.2f}"
            else: calc_str = str(calc)
            
            abs_str = f"{abs_d:.2f}"
            pct_str = f"{pct_d:.1f}%"
            
            # Highlight bad results with bold
            is_bad = False
            if metric == 'total_beats' and abs(pct_d) > 1.0: is_bad = True
            elif 'hr' in metric and abs(abs_d) > 5: is_bad = True
            elif abs(pct_d) > 20.0 and abs(abs_d) > 1: is_bad = True # Only flag large % if abs diff is also meaningful
            
            if is_bad:
                pct_str = f"**{pct_str}**"
                abs_str = f"**{abs_str}**"
            
            md_lines.append(f"| {name} | {gt_str} | {calc_str} | {abs_str} | {pct_str} |")
            
        md_lines.append("")
        
    # Write file
    with open(output_path, 'w') as f:
        f.write('\n'.join(md_lines))
    
    print(f"Markdown report generated at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Markdown Report from JSON Validation Results")
    parser.add_argument('json_file', nargs='?', help='Path to the input JSON validation report')
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if args.json_file:
        JSON_PATH = args.json_file
        # Derive output path from input path
        base_name = os.path.splitext(JSON_PATH)[0]
        MD_PATH = f"{base_name}.md"
    else:
        # Default behavior
        JSON_PATH = os.path.join(BASE_DIR, 'validation_results', 'validation_report.json')
        MD_PATH = os.path.join(BASE_DIR, 'validation_results', 'validation_report.md')
    
    if not os.path.exists(JSON_PATH):
        print(f"Error: Input file not found: {JSON_PATH}")
        exit(1)

    print(f"Input JSON: {JSON_PATH}")
    print(f"Output MD:  {MD_PATH}")
    
    generate_markdown_report(JSON_PATH, MD_PATH)
