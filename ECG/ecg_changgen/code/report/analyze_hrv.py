import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import argparse
import os

def analyze_hrv(record_name, fs=125):
    print(f"Starting HRV Analysis for record: {record_name}")
    
    # 1. Load Data
    csv_path = f'input/{record_name}_predictions.csv'
    if not os.path.exists(csv_path):
        print(f"Error: Predictions file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Extract Peaks and Classes
    # R_Peak_Pos is in samples (assuming 125Hz based on previous context)
    r_peaks = df['R_Peak_Pos'].values
    classes = df['Predicted_Class'].values # 0=N, 1=S, 2=V...
    
    print(f"Loaded {len(r_peaks)} beats.")
    
    # --- Arrhythmia Pattern Detection (Bigeminy/Trigeminy) ---
    print("\n--- Arrhythmia Pattern Detection ---")
    
    # Map classes for pattern matching:
    # 0 -> N (Normal)
    # 1 -> S (Supraventricular)
    # 2 -> V (Ventricular)
    # 3 -> F (Fusion) -> Treat as N for rhythm or V depending on dominance? Usually similar to V in context of ectopy. Let's treat F and Q as Non-N for generic ectopy, or specifically V.
    # For standard Bigeminy, we look for N-V-N-V or N-S-N-S.
    
    # Let's simplify to: 0=N, Other=E (Ectopic)
    # Strict Bigeminy: N-E-N-E-N-E
    # Strict Trigeminy: N-N-E-N-N-E
    
    # We will look for sequences.
    # N=0, V=2. Let's focus on Ventricular Bigeminy (N-V-N-V) as it's most common clinically.
    
    # Create a simplified symbol array
    # 'N' for Class 0
    # 'V' for Class 2 (Ventricular)
    # 'S' for Class 1 (Supraventricular)
    # 'O' for Others
    
    symbols = np.array(['O'] * len(classes))
    symbols[classes == 0] = 'N'
    symbols[classes == 1] = 'S'
    symbols[classes == 2] = 'V'
    
    symbol_str = "".join(symbols)
    
    # Helper to count patterns
    def count_episodes(pattern_name, pattern_regex, min_repeats=3):
        import re
        # Pattern for Bigeminy (NV): (NV){3,}
        # Pattern for Trigeminy (NNV): (NNV){3,}
        
        # We need to construct regex.
        # e.g. (NV){3,} matches NVNVNV or more.
        
        regex = f"({pattern_regex}){{{min_repeats},}}"
        matches = list(re.finditer(regex, symbol_str))
        
        count = len(matches)
        total_beats = sum([len(m.group(0)) for m in matches])
        max_len_beats = max([len(m.group(0)) for m in matches]) if matches else 0
        
        print(f"{pattern_name}: {count} episodes")
        if count > 0:
            print(f"  Total beats involved: {total_beats}")
            print(f"  Longest episode: {max_len_beats} beats")
            # print(f"  Locations (first 3): {[m.start() for m in matches[:3]]}")
            
    # Ventricular Bigeminy: N-V
    count_episodes("Ventricular Bigeminy (N-V)", "NV")
    
    # Ventricular Trigeminy: N-N-V
    count_episodes("Ventricular Trigeminy (N-N-V)", "NNV")
    
    # Supraventricular Bigeminy: N-S
    count_episodes("Supraventricular Bigeminy (N-S)", "NS")
    
    # Supraventricular Trigeminy: N-N-S
    count_episodes("Supraventricular Trigeminy (N-N-S)", "NNS")

    # 2. Compute RR Intervals (in ms)
    # RR[i] is the interval between peak[i] and peak[i+1]
    rr_intervals = np.diff(r_peaks) / fs * 1000
    
    # 3. Identify NN Intervals (Normal-to-Normal)
    # Strict definition: 
    # - Both start and end beats are Normal (Class 0)
    # - Interval is within physiological limits (300ms - 2000ms) to exclude artifacts
    
    is_normal = (classes == 0)
    # Initial NN mask based on classification
    nn_mask = is_normal[:-1] & is_normal[1:]
    
    # Apply physiological filter to the mask
    # We want to exclude artifacts from the "Cleaned NN" analysis
    is_physiological = (rr_intervals >= 300) & (rr_intervals <= 2000)
    nn_mask = nn_mask & is_physiological
    
    nn_intervals = rr_intervals[nn_mask]
    
    print(f"Total RR Intervals: {len(rr_intervals)}")
    print(f"Valid NN Intervals (Normal & Physiological): {len(nn_intervals)} ({len(nn_intervals)/len(rr_intervals)*100:.1f}%)")

    # --- Data Quality Check ---
    print("\n--- Data Quality Check ---")
    rr_min = np.min(rr_intervals)
    rr_max = np.max(rr_intervals)
    print(f"RR Interval Range: {rr_min:.2f} ms - {rr_max:.2f} ms")
    
    # Check for physiological outliers (e.g. < 300ms or > 2000ms)
    n_short = np.sum(rr_intervals < 300)
    n_long = np.sum(rr_intervals > 2000)
    if n_short > 0 or n_long > 0:
        print(f"Warning: Found {n_short} extremely short (<300ms) and {n_long} extremely long (>2000ms) intervals.")
        print("These might be artifacts or ectopic beats affecting HRV analysis.")

    # Limit data size for memory safety if too large
    MAX_BEATS = 5000
    if len(r_peaks) > MAX_BEATS:
        print(f"\n[Warning] Data too large ({len(r_peaks)} beats). Limiting to first {MAX_BEATS} beats for detailed NeuroKit2 analysis to avoid OOM.")
        print("Calculating full Time-Domain metrics manually below...")
        analysis_peaks = r_peaks[:MAX_BEATS]
    else:
        analysis_peaks = r_peaks

    # 4. NeuroKit2 Analysis
    # We will use nk.hrv() which provides a comprehensive summary.
    # Note: nk.hrv takes 'peaks' indices. 
    # By default, we run it on ALL peaks to see the overall variability (including arrhythmia).
    # Then we can manually compute cleaner metrics or try to pass cleaned peaks.
    
    print("\n--- Generating HRV Report (NeuroKit2) ---")
    try:
        # Generate plot and metrics
        # nk.hrv automatically performs Time, Frequency, and Non-Linear analysis
        hrv_metrics = nk.hrv(peaks=analysis_peaks, sampling_rate=fs, show=True)
        
        # Save the plot
        plot_path = f'input/{record_name}_hrv_plot.png'
        # nk.hrv(show=True) creates a figure, we can save it.
        # Get current figure
        fig = plt.gcf()
        fig.set_size_inches(15, 10)
        plt.savefig(plot_path, dpi=150)
        print(f"HRV visualization saved to: {plot_path}")
        
        # Display key metrics from NeuroKit2 (Full Signal)
        print("\n[NeuroKit2 - Raw Signal Metrics]")
        cols_to_show = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_LF', 'HRV_HF', 'HRV_LFHF']
        for col in cols_to_show:
            if col in hrv_metrics.columns:
                # Use higher precision to detect small values (e.g. if units are s^2 instead of ms^2)
                val = hrv_metrics[col].values[0]
                print(f"{col}: {val:.6f}")
                
    except Exception as e:
        print(f"Error running NeuroKit2 HRV: {e}")

    # 5. Manual Clean NN Analysis (Time Domain)
    # Since NeuroKit2 might be sensitive to the full signal's irregularities (like V-beats),
    # we manually compute Time-Domain metrics on the strictly filtered NN intervals.
    print("\n--- Cleaned NN Analysis (Manual Calculation) ---")
    if len(nn_intervals) > 1:
        # Mean NN and SDNN use all valid NN intervals (descriptive stats)
        mean_nn = np.mean(nn_intervals)
        sdnn = np.std(nn_intervals, ddof=1)
        
        # RMSSD and pNN50 require ADJACENT NN intervals.
        # If we just take diff(nn_intervals), we might subtract two intervals that are miles apart 
        # (if the beats between them were removed).
        # We must ensure continuity.
        
        # nn_mask[i] is True means interval i (from beat i to i+1) is valid.
        # nn_mask[i+1] is True means interval i+1 (from beat i+1 to i+2) is valid.
        # If both are True, then interval i and interval i+1 are physically adjacent.
        
        adjacent_mask = nn_mask[:-1] & nn_mask[1:]
        
        # We need the differences between adjacent RR intervals
        # all_diffs[i] = rr_intervals[i+1] - rr_intervals[i]
        # We only keep all_diffs[i] if adjacent_mask[i] is True
        
        all_diffs = np.diff(rr_intervals)
        valid_diffs = all_diffs[adjacent_mask]
        
        if len(valid_diffs) > 0:
            rmssd = np.sqrt(np.mean(valid_diffs**2))
            pnn50 = np.sum(np.abs(valid_diffs) > 50) / len(valid_diffs) * 100
            
            # --- Jump Analysis (Diagnostic) ---
            print("\n--- Diagnostic: Beat-to-Beat Jumps ---")
            jumps_abs = np.abs(valid_diffs)
            print(f"Max Jump: {np.max(jumps_abs):.2f} ms")
            print(f"Jumps > 100ms: {np.sum(jumps_abs > 100)} ({np.sum(jumps_abs > 100)/len(valid_diffs)*100:.1f}%)")
            print(f"Jumps > 300ms: {np.sum(jumps_abs > 300)} ({np.sum(jumps_abs > 300)/len(valid_diffs)*100:.1f}%)")
            
            if np.sum(jumps_abs > 100) / len(valid_diffs) > 0.05:
                print("Warning: High percentage of large jumps detected in 'Normal' beats.")
                print("This suggests contamination by arrhythmias (e.g. AFib) or misclassified ectopic beats.")

            # --- Strict Filtering (20% Rule) ---
            # Exclude pairs where change is > 20% of previous interval
            # This simulates a stricter "Sinus Rhythm" filter
            
            # Re-calculate indices to access original values
            # adjacent_indices = np.where(adjacent_mask)[0]
            # prev_intervals = rr_intervals[adjacent_indices]
            # next_intervals = rr_intervals[adjacent_indices+1]
            
            # Or simpler:
            # We already have valid_diffs. We need the base interval to compare percentage.
            # valid_diffs[k] corresponds to difference at k-th adjacent pair.
            # We need the actual interval values for those pairs.
            
            # Let's get the values of the "first" interval in each adjacent pair
            pair_starts = rr_intervals[:-1][adjacent_mask]
            
            pct_change = np.abs(valid_diffs) / pair_starts
            strict_mask = pct_change <= 0.20
            
            valid_diffs_strict = valid_diffs[strict_mask]
            
            if len(valid_diffs_strict) > 0:
                rmssd_strict = np.sqrt(np.mean(valid_diffs_strict**2))
                pnn50_strict = np.sum(np.abs(valid_diffs_strict) > 50) / len(valid_diffs_strict) * 100
                print("\n--- Strictly Filtered Analysis (Exclude jumps > 20%) ---")
                print(f"Retained Pairs: {len(valid_diffs_strict)} ({len(valid_diffs_strict)/len(valid_diffs)*100:.1f}%)")
                print(f"RMSSD (Strict): {rmssd_strict:.2f} ms")
                print(f"pNN50 (Strict): {pnn50_strict:.2f} %")
            
        else:
            rmssd = 0.0
            pnn50 = 0.0
        
        print("\n--- Standard Cleaned NN Metrics ---")
        print(f"Mean NN: {mean_nn:.2f} ms")
        print(f"SDNN:    {sdnn:.2f} ms")
        print(f"RMSSD:   {rmssd:.2f} ms (calculated on {len(valid_diffs)} adjacent pairs)")
        print(f"pNN50:   {pnn50:.2f} %")
    else:
        print("Not enough NN intervals for analysis.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ECG HRV Analysis')
    parser.add_argument('--record', type=str, default='16265', help='Record name')
    args = parser.parse_args()
    
    analyze_hrv(args.record)
