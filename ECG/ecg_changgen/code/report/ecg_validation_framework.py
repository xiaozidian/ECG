import os
import json
import pandas as pd
import numpy as np
import re
from datetime import timedelta
from scipy.signal import welch, detrend
import glob
import sys
import argparse

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ECGValidator:
    def __init__(self, ground_truth_path, data_dir, output_dir='validation_output', model_path=None):
        self.ground_truth_path = ground_truth_path
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.model_path = model_path
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.ground_truth = self._load_ground_truth()
        
    def _load_ground_truth(self):
        with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def load_data(self, record_id):
        """
        Finds the .DATA file for a given record ID.
        Searches recursively in data_dir.
        """
        # Pattern: data_dir/**/record_id.DATA
        # Using glob to find the file
        search_pattern = os.path.join(self.data_dir, '**', f'{record_id}.DATA')
        files = glob.glob(search_pattern, recursive=True)
        
        if not files:
            # Try searching just by filename in case structure is flat or different
            search_pattern = os.path.join(self.data_dir, f'{record_id}.DATA')
            files = glob.glob(search_pattern)
            
        if files:
            return files[0]
        return None

    def process_data(self, record_id):
        """
        Process the data using clinical_processor to get predictions.csv.
        Returns the path to the predictions CSV.
        """
        data_path = self.load_data(record_id)
        if not data_path:
            print(f"Error: Data file for record {record_id} not found.")
            return None
            
        # Check if predictions already exist to save time
        # clinical_processor outputs to output_dir/{record_name}_predictions.csv
        # But wait, clinical_processor.process_clinical_record takes output_dir as arg.
        # Let's use self.output_dir/processed/{record_id}
        
        proc_output_dir = os.path.join(self.output_dir, 'processed', record_id)
        if not os.path.exists(proc_output_dir):
            os.makedirs(proc_output_dir)
            
        csv_path = os.path.join(proc_output_dir, f'{record_id}_predictions.csv')
        
        if os.path.exists(csv_path):
            print(f"Predictions found for {record_id}, skipping processing.")
            return csv_path
            
        print(f"Processing record {record_id}...")
        try:
            import clinical_processor
            # We need to find where the model file is. 
            if self.model_path and os.path.exists(self.model_path):
                model_path = self.model_path
            else:
                # Fallback to default search logic
                # Assuming it's in the project root or same dir as clinical_processor
                model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'clinical_cnn_mitbih.h5')
                if not os.path.exists(model_path):
                     # Try current directory or code directory
                     model_path = 'clinical_cnn_mitbih.h5'
                     if not os.path.exists(model_path):
                         model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clinical_cnn_mitbih.h5')
            
            result_path = clinical_processor.process_clinical_record(
                data_path, 
                output_dir=proc_output_dir,
                model_path=model_path
            )
            return result_path
        except Exception as e:
            print(f"Error processing {record_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_metrics(self, csv_path, fs=500):
        """
        Calculates metrics from predictions CSV.
        Replicates logic from generate_clinical_report.py
        """
        if not csv_path or not os.path.exists(csv_path):
            return None
            
        df = pd.read_csv(csv_path)
        r_peaks = df['R_Peak_Pos'].values
        classes = df['Predicted_Class'].values
        
        # --- Basic Stats ---
        total_beats = len(r_peaks)
        
        # RR Intervals (ms)
        rr_intervals = np.diff(r_peaks) / fs * 1000
        rr_times = r_peaks[1:] / fs # in seconds
        
        # --- Summary Statistics ---
        valid_hr_mask = (rr_intervals >= 300) & (rr_intervals <= 2000)
        valid_rr = rr_intervals[valid_hr_mask]
        valid_rr_times = rr_times[valid_hr_mask]
        
        avg_hr = 0
        min_hr = 0
        max_hr = 0
        max_rr = 0
        
        if len(valid_rr) > 0:
            avg_hr = float(np.mean(60000 / valid_rr))
            
            # HR Cleaning logic from generate_clinical_report.py
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
            
            # Adaptive window for smoothing
            median_rr_s = float(np.median(hr_rr) / 1000.0) if len(hr_rr) > 0 else 1.0
            
            # Calculate Min HR (smoothed)
            window_long = int(round(60.0 / median_rr_s)) if median_rr_s > 0 else 60
            window_long = max(4, min(window_long, 120))
            hr_series = pd.Series(inst_hr)
            smooth_hr_min = hr_series.rolling(window=window_long, center=True).mean().dropna()
            min_hr = float(np.min(smooth_hr_min)) if len(smooth_hr_min) > 0 else (float(np.min(inst_hr)) if len(inst_hr) > 0 else 0.0)

            rr_series_for_max = pd.Series(valid_rr)
            rr_med_for_max = rr_series_for_max.rolling(window=11, center=True, min_periods=6).median()
            rr_med_for_max = rr_med_for_max.fillna(float(np.median(valid_rr)))
            hr_clean_mask_for_max = (np.abs(rr_series_for_max - rr_med_for_max) <= (0.35 * rr_med_for_max)).values

            hr_rr_for_max = valid_rr[hr_clean_mask_for_max]
            if len(hr_rr_for_max) < 4:
                hr_rr_for_max = valid_rr

            inst_hr_for_max = 60000 / hr_rr_for_max
            median_rr_s_for_max = float(np.median(hr_rr_for_max) / 1000.0) if len(hr_rr_for_max) > 0 else median_rr_s

            window_short = int(round(6.0 / median_rr_s_for_max)) if median_rr_s_for_max > 0 else 6
            window_short = max(3, min(window_short, 60))

            hr_series_for_max = pd.Series(inst_hr_for_max)
            smooth_hr_max_series = hr_series_for_max.rolling(window=window_short, center=True).mean().dropna()
            if len(smooth_hr_max_series) == 0:
                smooth_hr_max_series = hr_series_for_max

            smooth_vals = smooth_hr_max_series.values
            if len(smooth_vals) == 0:
                max_hr = float(np.max(inst_hr)) if len(inst_hr) > 0 else 0.0
            else:
                top_n = min(200, len(smooth_vals))
                top_idx = np.argpartition(smooth_vals, -top_n)[-top_n:]
                top_idx = top_idx[np.argsort(smooth_vals[top_idx])[::-1]]

                chosen_max = None
                smooth_index = smooth_hr_max_series.index.values
                for i in top_idx:
                    idx = int(smooth_index[i])
                    start = max(0, idx - 20)
                    end = min(len(hr_rr_for_max), idx + 20)
                    seg = hr_rr_for_max[start:end]
                    if len(seg) < 10:
                        continue
                    seg_mean = float(np.mean(seg))
                    if seg_mean <= 0:
                        continue
                    cv = float(np.std(seg) / seg_mean)
                    if cv <= 0.04:
                        chosen_max = float(smooth_vals[i])
                        break

                max_hr = chosen_max if chosen_max is not None else float(np.max(smooth_vals))

        # Longest RR
        rr_long_mask = (rr_intervals >= 300) & (rr_intervals <= 3000)
        rr_long = rr_intervals[rr_long_mask]
        if len(rr_long) > 0:
            rr_long_series = pd.Series(rr_long)
            rr_long_med = rr_long_series.rolling(window=11, center=True, min_periods=6).median()
            rr_long_med = rr_long_med.fillna(float(np.median(rr_long)))
            rr_long_clean_mask = (np.abs(rr_long_series - rr_long_med) <= (0.35 * rr_long_med)).values
            rr_long_clean = rr_long[rr_long_clean_mask]
            if len(rr_long_clean) == 0:
                rr_long_clean = rr_long
            max_rr = float(np.max(rr_long_clean) / 1000.0)
        
        # --- Arrhythmia Analysis ---
        # 0: N, 1: S, 2: V, 3: F (Treat as V), 4: Q (Ignore/Other)
        
        # Simplify classes: 0=N, 1=S, 2=V (merge F into V)
        simple_classes = classes.copy()
        simple_classes[simple_classes == 3] = 2 # F -> V
        
        # Initialize counters
        v_counts = {'total': 0, 'single': 0, 'pair': 0, 'bigeminy': 0, 'trigeminy': 0, 'vt': 0}
        s_counts = {'total': 0, 'single': 0, 'pair': 0, 'bigeminy': 0, 'trigeminy': 0, 'svt': 0}
        
        def count_patterns(cls_code, counts_dict, run_key):
            # Mask for target class
            is_cls = (simple_classes == cls_code)
            counts_dict['total'] = int(np.sum(is_cls))
            
            # Find runs
            # We will iterate and count consecutive runs
            # This is a simple state machine approach
            run_len = 0
            runs = []
            
            for c in simple_classes:
                if c == cls_code:
                    run_len += 1
                else:
                    if run_len > 0:
                        runs.append(run_len)
                    run_len = 0
            if run_len > 0: runs.append(run_len)
            
            # Count singles, pairs, runs (>=3)
            counts_dict['single'] = runs.count(1)
            counts_dict['pair'] = runs.count(2)
            counts_dict[run_key] = sum(1 for r in runs if r >= 3)
            
            # Bigeminy/Trigeminy logic (Simplified)
            # Bigeminy: C, N, C, N...
            # Trigeminy: C, N, N, C, N, N...
            # We'll use a string representation for regex matching which is easier
            # Map classes to string: N->'N', S->'S', V->'V', Q->'Q'
            mapping = {0:'N', 1:'S', 2:'V', 4:'Q'}
            seq_str = "".join([mapping.get(c, 'Q') for c in simple_classes])
            target_char = mapping[cls_code]
            
            # Bigeminy: "VNVNV" (at least 3 repetitions of pattern?)
            # Usually definitions vary. Let's look for patterns.
            # Pattern: (V N)+ 
            # We can use regex to count occurrences of patterns.
            # Strictly, bigeminy is a rhythm, so it should persist for a while.
            # However, simpler count: look for isolated patterns?
            # Existing report just gives a count. Let's assume it counts individual events that fit the pattern.
            # But usually "Bigeminy: 4" means 4 episodes or 4 beats? 
            # Given the numbers in example (Total V: 5247, Bigeminy: 4), it likely means episodes.
            
            # Let's try to match patterns of "Target, Normal, Target"
            # Bigeminy pattern: T N T
            # Trigeminy pattern: T N N T
            
            # Note: This is a heuristic.
            # V Bigeminy: V N V N V...
            
            import re
            
            # Regex for Bigeminy: (V N){2,} V?
            # Let's count non-overlapping matches of "VNV" or similar?
            # Actually, standard Holter software counts "Bigeminy cycles".
            # A sequence V N V N V has 2 bigeminy cycles (VNV, VNV)? Or is it one episode?
            # Let's assume "Bigeminy" count refers to the number of *beats* involved or *episodes*?
            # Looking at the numbers: Total 5247. Single 5245. Pair 1 (2 beats). Bigeminy 4. Trigeminy 365.
            # 5245 + 1*2 = 5247. 
            # Wait, 5245 singles + 1 pair (2 beats) = 5247 total beats.
            # Then Bigeminy and Trigeminy must be subsets of these, or describing the rhythm context.
            # If Single + Pair*2 + Run*Length = Total, then Bigeminy/Trigeminy are tags.
            # In the example: 5245 (Single) + 1 (Pair, 2 beats) = 5247.
            # The sum matches exactly if Pair=1 means 1 pair (2 beats).
            # So Bigeminy (4) and Trigeminy (365) are likely categorizations of those Single beats.
            # i.e., A single V beat can be part of a Bigeminy pattern.
            
            # So, we should identify Single/Pair/Run first (which partitions the beats).
            # Then check if Single beats are part of Bigeminy/Trigeminy context.
            
            # Simple implementation:
            # 1. Bigeminy: V N V
            # 2. Trigeminy: V N N V
            
            # We will scan the sequence.
            bg_count = 0
            tg_count = 0
            
            # Convert to list for easier indexing
            s_cls = list(simple_classes)
            n = len(s_cls)
            
            # We iterate and check context.
            # This is slow in Python but N ~ 100k, so it's fine (O(N)).
            
            for i in range(1, n-1):
                if s_cls[i] == cls_code:
                    # Check for Bigeminy context: Prev=N, Next=N (Isolated V)
                    # And check if it forms a pattern with neighbors.
                    
                    # Look ahead for Bigeminy: V N V
                    if i+2 < n and s_cls[i+1] == 0 and s_cls[i+2] == cls_code:
                        bg_count += 1
                        
                    # Look ahead for Trigeminy: V N N V
                    if i+3 < n and s_cls[i+1] == 0 and s_cls[i+2] == 0 and s_cls[i+3] == cls_code:
                        tg_count += 1
                        
            # Note: This counts "intervals" of bigeminy/trigeminy.
            # e.g. V N V N V -> 2 bigeminy intervals.
            # This seems reasonable.
            
            counts_dict['bigeminy'] = bg_count
            counts_dict['trigeminy'] = tg_count

        count_patterns(2, v_counts, 'vt') # V
        count_patterns(1, s_counts, 'svt') # S

        
        # --- HRV Analysis ---
        is_physiological = (rr_intervals >= 300) & (rr_intervals <= 2000)
        rr_phys = rr_intervals[is_physiological]
        rr_phys_times = rr_times[is_physiological]
        rr_phys_idx = np.where(is_physiological)[0]
        
        sdnn = sdann = rmssd = pnn50 = tri_index = lf = hf = lfhf = 0
        
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
            
        if len(clean_nn) > 1:
            sdnn = np.std(clean_nn, ddof=1)
            
            consecutive = np.diff(nn_idx) == 1
            all_diffs = np.diff(clean_nn)
            valid_diffs = all_diffs[consecutive]
            
            if len(valid_diffs) > 0:
                rmssd = np.sqrt(np.mean(valid_diffs**2))
                pnn50 = np.sum(np.abs(valid_diffs) > 50) / len(valid_diffs) * 100
                
            # SDANN
            if len(nn_times) > 0:
                segment_duration = 5 * 60
                start_time = nn_times[0]
                end_time = nn_times[-1]
                num_segments = int((end_time - start_time) / segment_duration)
                
                segment_means = []
                for i in range(num_segments + 1):
                    seg_start = start_time + i * segment_duration
                    seg_end = seg_start + segment_duration
                    in_segment = (nn_times >= seg_start) & (nn_times < seg_end)
                    segment_nns = clean_nn[in_segment]
                    if len(segment_nns) > 30:
                        segment_means.append(np.mean(segment_nns))
                
                if len(segment_means) > 1:
                    sdann = np.std(segment_means, ddof=1)
                    
            # Triangular Index
            if len(clean_nn) > 0:
                bins = np.arange(np.min(clean_nn), np.max(clean_nn) + 8, 8)
                hist, _ = np.histogram(clean_nn, bins=bins)
                max_density = np.max(hist)
                if max_density > 0:
                    tri_index = len(clean_nn) / max_density
                    
            # Frequency Domain
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

                        if len(seg_nn) < 60: continue
                        if (seg_times[-1] - seg_times[0]) < (segment_duration * 0.8): continue

                        t = seg_times - seg_start
                        t_interp = np.arange(0, segment_duration, 1 / fs_interp)
                        rr_interp = np.interp(t_interp, t, seg_nn)
                        rr_interp = detrend(rr_interp, type='linear')

                        nperseg = min(256, len(rr_interp))
                        if nperseg < 64: continue

                        f, pxx = welch(rr_interp, fs=fs_interp, nperseg=nperseg)
                        
                        # Correct s^2 to ms^2 conversion (1e6)
                        # Input is already in ms, so no conversion needed if we want ms^2
                        # pxx = pxx * 1e6

                        lf_mask = (f >= 0.07) & (f < 0.15)
                        hf_mask = (f >= 0.15) & (f <= 0.4)

                        # Use np.trapz for compatibility with older numpy versions (e.g. 1.24)
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
                print(f"Warning: Frequency analysis failed: {e}")

        return {
            'total_beats': total_beats,
            'max_hr': max_hr,
            'min_hr': min_hr,
            'avg_hr': avg_hr,
            'max_rr': max_rr,
            'ventricular': v_counts,
            'supraventricular': s_counts,
            'sdnn': sdnn,
            'sdann': sdann,
            'rmssd': rmssd,
            'pnn50': pnn50,
            'lf': lf,
            'hf': hf,
            'lf_hf': lfhf,
            'tri_index': tri_index
        }

    def compare_metrics(self, ground_truth_record, calculated_metrics):
        """
        Compares ground truth metrics with calculated metrics.
        Returns a dict of differences.
        """
        gt_stats = ground_truth_record['overall_stats']
        gt_arr = ground_truth_record['arrhythmia_events']
        gt_hrv = ground_truth_record['hrv_metrics']
        
        diffs = {}
        
        # Helper to calculate % diff
        def calc_diff(key, gt_val, calc_val):
            abs_diff = calc_val - gt_val
            pct_diff = (abs_diff / gt_val * 100) if gt_val != 0 else 0
            return {'gt': gt_val, 'calc': calc_val, 'abs_diff': abs_diff, 'pct_diff': pct_diff}

        # Compare
        diffs['total_beats'] = calc_diff('total_beats', gt_stats['total_beats'], calculated_metrics['total_beats'])
        diffs['avg_hr'] = calc_diff('avg_hr', gt_stats['avg_hr'], calculated_metrics['avg_hr'])
        diffs['max_hr'] = calc_diff('max_hr', gt_stats['max_hr']['value'], calculated_metrics['max_hr'])
        diffs['min_hr'] = calc_diff('min_hr', gt_stats['min_hr']['value'], calculated_metrics['min_hr'])
        diffs['max_rr'] = calc_diff('max_rr', gt_stats['longest_rr']['value'], calculated_metrics['max_rr'])
        
        # Arrhythmia (using total counts)
        # Handle cases where GT might have "< 0.1" or similar strings for percentage, but we use counts here.
        # The JSON has integer totals.
        
        for evt_type in ['ventricular', 'supraventricular']:
            diffs[f'{evt_type}_total'] = calc_diff(f'{evt_type}_total', gt_arr[evt_type]['total'], calculated_metrics[evt_type]['total'])
            diffs[f'{evt_type}_single'] = calc_diff(f'{evt_type}_single', gt_arr[evt_type]['single'], calculated_metrics[evt_type]['single'])
            diffs[f'{evt_type}_pair'] = calc_diff(f'{evt_type}_pair', gt_arr[evt_type]['pair'], calculated_metrics[evt_type]['pair'])
            diffs[f'{evt_type}_bigeminy'] = calc_diff(f'{evt_type}_bigeminy', gt_arr[evt_type]['bigeminy'], calculated_metrics[evt_type]['bigeminy'])
            diffs[f'{evt_type}_trigeminy'] = calc_diff(f'{evt_type}_trigeminy', gt_arr[evt_type]['trigeminy'], calculated_metrics[evt_type]['trigeminy'])
            
            run_key = 'vt' if evt_type == 'ventricular' else 'svt'
            diffs[f'{evt_type}_{run_key}'] = calc_diff(f'{evt_type}_{run_key}', gt_arr[evt_type][run_key], calculated_metrics[evt_type][run_key])
        
        # HRV
        diffs['sdnn'] = calc_diff('sdnn', gt_hrv['sdnn'], calculated_metrics['sdnn'])
        diffs['sdann'] = calc_diff('sdann', gt_hrv['sdann'], calculated_metrics['sdann'])
        diffs['rmssd'] = calc_diff('rmssd', gt_hrv['rmssd'], calculated_metrics['rmssd'])
        diffs['pnn50'] = calc_diff('pnn50', gt_hrv['pnn50'], calculated_metrics['pnn50'])
        diffs['lf'] = calc_diff('lf', gt_hrv['lf'], calculated_metrics['lf'])
        diffs['hf'] = calc_diff('hf', gt_hrv['hf'], calculated_metrics['hf'])
        diffs['lf_hf'] = calc_diff('lf_hf', gt_hrv['lf_hf'], calculated_metrics['lf_hf'])
        diffs['tri_index'] = calc_diff('tri_index', gt_hrv['tri_index'], calculated_metrics['tri_index'])
        
        return diffs

    def run_validation(self):
        all_results = []
        
        for report in self.ground_truth['reports']:
            record_id = report['patient_info']['id']
            print(f"\n--- Validating Record: {record_id} ---")
            
            # 1. Process
            csv_path = self.process_data(record_id)
            if not csv_path:
                print(f"Skipping {record_id} due to processing failure.")
                continue
                
            # 2. Calculate Metrics
            calc_metrics = self.calculate_metrics(csv_path)
            if not calc_metrics:
                print(f"Skipping {record_id} due to metric calculation failure.")
                continue
                
            # 3. Compare
            diffs = self.compare_metrics(report, calc_metrics)
            
            result = {
                'record_id': record_id,
                'diffs': diffs
            }
            all_results.append(result)
            
            # Print immediate summary for this record
            print(f"Comparison Summary for {record_id}:")
            print(f"{'Metric':<25} {'GT':<10} {'Calc':<10} {'Diff':<10} {'Diff%':<10}")
            for k, v in diffs.items():
                print(f"{k:<25} {v['gt']:<10.2f} {v['calc']:<10.2f} {v['abs_diff']:<10.2f} {v['pct_diff']:<10.1f}%")

        self.generate_report(all_results)
        
    def generate_report(self, results):
        model_name = "default"
        if self.model_path:
             model_name = os.path.splitext(os.path.basename(self.model_path))[0]
        
        report_filename = f'validation_report_{model_name}.json'
        report_path = os.path.join(self.output_dir, report_filename)
        
        # Convert numpy types to native types for JSON serialization
        def convert(o):
            if isinstance(o, np.generic): return o.item()
            raise TypeError
            
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=convert)
            
        print(f"\nValidation complete. Full report saved to {report_path}")
        
        # Aggregate stats (MAE)
        print("\n=== Overall Performance (MAE) ===")
        metrics = results[0]['diffs'].keys() if results else []
        mae = {m: [] for m in metrics}
        
        for r in results:
            for m in metrics:
                mae[m].append(abs(r['diffs'][m]['abs_diff']))
                
        for m, diffs in mae.items():
            if diffs:
                print(f"{m:<25}: {np.mean(diffs):.2f}")

def _load_ground_truth_record(ground_truth_path, record_id):
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        gt = json.load(f)
    for report in gt.get('reports', []):
        pid = (report.get('patient_info') or {}).get('id')
        if pid == record_id:
            return report
    return None

def read_ebi_annotations(ebi_path, data_path=None, num_channels=3, fs=500):
    import struct

    with open(ebi_path, 'rb') as f:
        raw = f.read()

    if len(raw) < 32:
        raise ValueError(f"EBI file too small: {ebi_path}")

    magic, record_len, v1, v2, ts = struct.unpack('<5I', raw[:20])
    header_len = 32
    if record_len % 4 != 0:
        raise ValueError(f"Unexpected record_len in EBI: {record_len}")

    if (len(raw) - header_len) % record_len != 0:
        raise ValueError(f"EBI size not aligned to record_len: {ebi_path}")

    n = (len(raw) - header_len) // record_len
    cols = record_len // 4
    arr = np.frombuffer(raw, dtype=np.uint32, offset=header_len, count=n * cols).reshape(n, cols)
    if cols < 6:
        raise ValueError(f"Unexpected EBI record columns: {cols}")

    tick = arr[:, 0].astype(np.uint64)
    aux = arr[:, 1].astype(np.uint32)
    code = arr[:, 2].astype(np.uint32)
    beat_index = arr[:, 3].astype(np.uint32)
    rr_ms = arr[:, 5].astype(np.uint32)

    max_tick = int(tick.max()) if len(tick) else 0
    tick_seconds = 0.005
    if len(tick) > 10:
        dt_tick = np.diff(tick.astype(np.int64))
        rr_s = rr_ms.astype(np.float64) / 1000.0
        rr_s = rr_s[1:]
        m = (dt_tick > 0) & (rr_s > 0) & (rr_s < 5)
        if np.any(m):
            est = rr_s[m] / dt_tick[m].astype(np.float64)
            tick_seconds = float(np.median(est))
    if data_path and os.path.exists(data_path) and max_tick > 0 and (tick_seconds <= 0 or not np.isfinite(tick_seconds)):
        data_size = os.path.getsize(data_path)
        total_samples = data_size // (2 * num_channels)
        duration_s = total_samples / float(fs)
        tick_seconds = float(duration_s) / float(max_tick)

    time_s = tick.astype(np.float64) * float(tick_seconds)
    sample_index = np.rint(time_s * float(fs)).astype(np.int64)
    aux_s = aux == np.uint32(131072)
    is_s = aux_s & ((code & np.uint32(536870912)) == 0)
    is_n_from_aux = aux_s & ~is_s
    is_v = (aux == np.uint32(196608)) | ((code & np.uint32(64)) != 0)
    is_n = aux == np.uint32(65536)
    label = np.full(len(code), 'Q', dtype=object)
    label[is_n | is_n_from_aux] = 'N'
    label[is_v] = 'V'
    label[is_s & ~is_v] = 'S'

    df = pd.DataFrame({
        'beat_index': beat_index,
        'tick': tick,
        'time_s': time_s,
        'sample_index': sample_index,
        'rr_ms': rr_ms,
        'code': code,
        'aux': aux,
        'label': label
    })

    meta = {
        'ebi_magic': int(magic),
        'ebi_record_len': int(record_len),
        'ebi_v1': int(v1),
        'ebi_v2': int(v2),
        'ebi_ts': int(ts),
        'tick_seconds': float(tick_seconds),
        'rows': int(len(df))
    }
    return df, meta

def read_pac_ticks(pac_path):
    import struct

    with open(pac_path, 'rb') as f:
        raw = f.read()

    if len(raw) < 32:
        raise ValueError(f"PAC file too small: {pac_path}")

    magic, record_len, v1, v2, ts = struct.unpack('<5I', raw[:20])
    header_len = 32
    if record_len <= 0 or record_len % 4 != 0:
        raise ValueError(f"Unexpected record_len in PAC: {record_len}")
    if (len(raw) - header_len) % record_len != 0:
        raise ValueError(f"PAC size not aligned to record_len: {pac_path}")

    n = (len(raw) - header_len) // record_len
    cols = record_len // 4
    arr = np.frombuffer(raw, dtype=np.uint32, offset=header_len, count=n * cols).reshape(n, cols)
    ticks = arr[:, 0].astype(np.uint64)
    meta = {
        'pac_magic': int(magic),
        'pac_record_len': int(record_len),
        'pac_v1': int(v1),
        'pac_v2': int(v2),
        'pac_ts': int(ts),
        'pac_rows': int(n)
    }
    return ticks, meta

def apply_pac_s_labels(ebi_df, pac_ticks, tolerance_ticks=0):
    ebi_ticks = ebi_df['tick'].to_numpy(dtype=np.int64)
    ebi_bi = ebi_df['beat_index'].to_numpy(dtype=np.int64)

    pac_ticks = np.asarray(pac_ticks, dtype=np.int64)
    idx = np.searchsorted(ebi_ticks, pac_ticks)
    idx0 = np.clip(idx, 0, len(ebi_ticks) - 1)
    idx1 = np.clip(idx - 1, 0, len(ebi_ticks) - 1)
    choose = np.where(np.abs(ebi_ticks[idx0] - pac_ticks) < np.abs(ebi_ticks[idx1] - pac_ticks), idx0, idx1)
    delta = np.abs(ebi_ticks[choose] - pac_ticks)
    if tolerance_ticks is None:
        dt = np.diff(ebi_ticks)
        dt = dt[dt > 0]
        median_dt = float(np.median(dt)) if len(dt) else 0.0
        p99 = float(np.percentile(delta, 99)) if len(delta) else 0.0
        if median_dt > 0 and np.isfinite(median_dt):
            tolerance_ticks = int(min(p99, 0.8 * median_dt))
        else:
            tolerance_ticks = int(p99)
        tolerance_ticks = max(0, int(tolerance_ticks))
    matched = choose[delta <= int(tolerance_ticks)]
    if len(matched) == 0:
        return ebi_df, {'s_beats': 0, 'tolerance_ticks': int(tolerance_ticks)}

    s_beat_index = np.unique(ebi_bi[matched])
    if 'label' not in ebi_df.columns:
        ebi_df['label'] = 'N'

    mask_s = ebi_df['beat_index'].isin(s_beat_index) & (ebi_df['label'] == 'N')
    ebi_df.loc[mask_s, 'label'] = 'S'
    return ebi_df, {'s_beats': int(mask_s.sum()), 'tolerance_ticks': int(tolerance_ticks)}

def convert_exam_to_csv(exam_path, output_csv_path=None, ground_truth_path=None):
    record_id = os.path.splitext(os.path.basename(exam_path))[0]
    record_root = os.path.dirname(os.path.dirname(exam_path))

    ebi_path = os.path.join(record_root, 'DGS', f'{record_id}.EBI')
    if not os.path.exists(ebi_path):
        matches = glob.glob(os.path.join(record_root, '**', f'{record_id}.EBI'), recursive=True)
        if matches:
            ebi_path = matches[0]
    if not os.path.exists(ebi_path):
        raise FileNotFoundError(f"EBI not found for record {record_id}: {ebi_path}")

    data_path = os.path.join(record_root, 'data', f'{record_id}.DATA')
    df, meta = read_ebi_annotations(ebi_path, data_path=data_path)

    gt_s_total = None
    if ground_truth_path and os.path.exists(ground_truth_path):
        report = _load_ground_truth_record(ground_truth_path, record_id)
        record_time = None
        if report:
            record_time = (report.get('patient_info') or {}).get('record_time')
            gt_s_total = ((report.get('arrhythmia_events') or {}).get('supraventricular') or {}).get('total')
        if record_time:
            base = pd.to_datetime(record_time)
            df.insert(3, 'timestamp', (base + pd.to_timedelta(df['time_s'], unit='s')).dt.strftime('%Y-%m-%d %H:%M:%S'))

    pac_path = os.path.join(record_root, 'data', f'{record_id}.PAC')
    if os.path.exists(pac_path):
        current_s = int((df['label'] == 'S').sum())
        need_pac = gt_s_total is None or (isinstance(gt_s_total, (int, float)) and current_s < int(gt_s_total))
        if need_pac:
            pac_ticks, pac_meta = read_pac_ticks(pac_path)
            df, pac_apply_meta = apply_pac_s_labels(df, pac_ticks, tolerance_ticks=None)
            meta.update(pac_meta)
            meta.update(pac_apply_meta)

    if output_csv_path is None:
        output_csv_path = os.path.join(os.path.dirname(exam_path), f'{record_id}_beat_annotations.csv')

    df.to_csv(output_csv_path, index=False)
    return output_csv_path, df, meta

def export_real_data_to_mitbih_like_csv(real_data_dir, train_data_dir, ground_truth_path=None, test_ratio=0.2, seed=42):
    import neurokit2 as nk
    from scipy.stats import kurtosis

    data_paths = glob.glob(os.path.join(real_data_dir, '**', 'data', '*.DATA'), recursive=True)
    record_ids = sorted([os.path.splitext(os.path.basename(p))[0] for p in data_paths])
    if not record_ids:
        raise FileNotFoundError(f"No .DATA files found under: {real_data_dir}")

    rng = np.random.default_rng(int(seed))
    record_ids_shuffled = record_ids.copy()
    rng.shuffle(record_ids_shuffled)
    n_test = max(1, int(round(len(record_ids_shuffled) * float(test_ratio))))
    test_set = set(record_ids_shuffled[:n_test])

    os.makedirs(train_data_dir, exist_ok=True)
    train_csv = os.path.join(train_data_dir, 'clinical_mitbih_train.csv')
    test_csv = os.path.join(train_data_dir, 'clinical_mitbih_test.csv')
    for p in [train_csv, test_csv]:
        if os.path.exists(p):
            os.remove(p)

    label_map = {'N': 0.0, 'S': 1.0, 'V': 2.0}

    def best_channel(data, limit=300000):
        lim = min(len(data), int(limit))
        scores = []
        for i in range(data.shape[1]):
            scores.append(float(kurtosis(data[:lim, i], fisher=False, bias=False)))
        return int(np.argmax(scores))

    def append_rows(csv_path, arr):
        if arr.size == 0:
            return
        with open(csv_path, 'a') as f:
            np.savetxt(f, arr, delimiter=',', fmt='%.8f')

    total_written = {'train': 0, 'test': 0}

    for record_id in record_ids:
        record_root = os.path.join(real_data_dir, record_id)
        exam_path = os.path.join(record_root, 'data', f'{record_id}.EXAM')
        data_path = os.path.join(record_root, 'data', f'{record_id}.DATA')
        if not os.path.exists(exam_path) or not os.path.exists(data_path):
            continue

        _, ann_df, _ = convert_exam_to_csv(exam_path, ground_truth_path=ground_truth_path)
        ann_df = ann_df[ann_df['label'].isin(list(label_map.keys()))].copy()
        ann_df = ann_df[ann_df['rr_ms'] > 0]
        if len(ann_df) == 0:
            continue

        with open(data_path, 'rb') as f:
            raw = f.read()
        data_int16 = np.frombuffer(raw, dtype=np.int16)
        remainder = len(data_int16) % 3
        if remainder != 0:
            data_int16 = data_int16[:-remainder]
        data = data_int16.reshape(-1, 3)
        ch = best_channel(data)
        signal = data[:, ch].astype(np.float64)

        fs_original = 500
        ecg_cleaned_high = nk.ecg_clean(signal, sampling_rate=fs_original, method='neurokit')
        target_fs = 125
        ecg_cleaned = nk.signal_resample(ecg_cleaned_high, sampling_rate=fs_original, desired_sampling_rate=target_fs)

        idx_125 = np.rint(ann_df['sample_index'].to_numpy(dtype=np.float64) * (target_fs / fs_original)).astype(np.int64)
        start = idx_125 - 50
        end = idx_125 + 137
        valid = (start >= 0) & (end < len(ecg_cleaned))
        if not np.any(valid):
            continue

        idx_125 = idx_125[valid]
        labels = ann_df['label'].to_numpy()[valid]
        labels = np.array([label_map[str(x)] for x in labels], dtype=np.float32)

        offsets = np.arange(-50, 137, dtype=np.int64)
        gather_idx = idx_125[:, None] + offsets[None, :]
        segs = ecg_cleaned[gather_idx]
        segs = segs.astype(np.float32)
        seg_min = segs.min(axis=1, keepdims=True)
        seg_max = segs.max(axis=1, keepdims=True)
        denom = seg_max - seg_min
        denom[denom == 0] = 1.0
        segs = (segs - seg_min) / denom

        out = np.concatenate([segs, labels[:, None]], axis=1)
        if record_id in test_set:
            append_rows(test_csv, out)
            total_written['test'] += int(out.shape[0])
        else:
            append_rows(train_csv, out)
            total_written['train'] += int(out.shape[0])

    return {
        'train_csv': train_csv,
        'test_csv': test_csv,
        'train_rows': int(total_written['train']),
        'test_rows': int(total_written['test']),
        'record_ids': record_ids,
        'test_record_ids': sorted(list(test_set)),
    }

def export_all_beat_annotation_csv(real_data_dir, ground_truth_path, overwrite=True):
    reports = []
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        gt = json.load(f)
    for r in gt.get('reports', []):
        rid = (r.get('patient_info') or {}).get('id')
        if not rid:
            continue
        exam_path = os.path.join(real_data_dir, rid, 'data', f'{rid}.EXAM')
        if not os.path.exists(exam_path):
            continue
        out_csv = os.path.join(os.path.dirname(exam_path), f'{rid}_beat_annotations.csv')
        if os.path.exists(out_csv) and not overwrite:
            reports.append({'record_id': rid, 'csv_path': out_csv})
            continue
        csv_path, df, meta = convert_exam_to_csv(exam_path, output_csv_path=out_csv, ground_truth_path=ground_truth_path)
        counts = df['label'].value_counts().to_dict() if 'label' in df.columns else {}
        reports.append({'record_id': rid, 'csv_path': csv_path, 'label_counts': counts, 'meta': meta})
    return reports

def verify_beat_annotation_against_all_report(ground_truth_path, beat_annotation_reports):
    by_id = {x.get('record_id'): x for x in beat_annotation_reports if x.get('record_id')}
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        gt = json.load(f)

    results = []
    for r in gt.get('reports', []):
        rid = (r.get('patient_info') or {}).get('id')
        if not rid:
            continue
        rep = by_id.get(rid)
        if not rep:
            results.append({'record_id': rid, 'status': 'missing_csv'})
            continue
        csv_path = rep.get('csv_path')
        if not csv_path or not os.path.exists(csv_path):
            results.append({'record_id': rid, 'status': 'missing_csv'})
            continue

        df = pd.read_csv(csv_path)
        nsv = df[df['label'].isin(['N', 'S', 'V'])].copy()
        n_count = int((nsv['label'] == 'N').sum())
        s_count = int((nsv['label'] == 'S').sum())
        v_count = int((nsv['label'] == 'V').sum())
        total = int(len(nsv))

        gt_total = int((r.get('overall_stats') or {}).get('total_beats') or 0)
        gt_v = int((((r.get('arrhythmia_events') or {}).get('ventricular') or {}).get('total')) or 0)
        gt_s = int((((r.get('arrhythmia_events') or {}).get('supraventricular') or {}).get('total')) or 0)
        gt_n = gt_total - gt_v - gt_s

        results.append({
            'record_id': rid,
            'status': 'ok' if (total == gt_total and v_count == gt_v and s_count == gt_s and n_count == gt_n) else 'mismatch',
            'csv_path': csv_path,
            'gt': {'total': gt_total, 'N': gt_n, 'S': gt_s, 'V': gt_v},
            'csv': {'total': total, 'N': n_count, 'S': s_count, 'V': v_count},
            'diff': {'total': total - gt_total, 'N': n_count - gt_n, 'S': s_count - gt_s, 'V': v_count - gt_v},
        })
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG Validation Framework")
    parser.add_argument('--model', type=str, default=None, help='Path to the model file (.h5)')
    parser.add_argument('--ground_truth', type=str, default=None, help='Path to ground truth JSON')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to output directory')
    
    args = parser.parse_args()

    # Configuration
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Assuming code/ecg_validation_framework.py, so project root is one level up
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    
    GROUND_TRUTH_PATH = args.ground_truth if args.ground_truth else os.path.join(PROJECT_ROOT, 'all_report.json')
    DATA_DIR = args.data_dir if args.data_dir else os.path.join(PROJECT_ROOT, 'real_data')
    OUTPUT_DIR = args.output_dir if args.output_dir else os.path.join(PROJECT_ROOT, 'validation_results')
    MODEL_PATH = args.model
    
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Ground Truth: {GROUND_TRUTH_PATH}")
    print(f"Data Dir: {DATA_DIR}")
    if MODEL_PATH:
        print(f"Model Path: {MODEL_PATH}")
    
    validator = ECGValidator(GROUND_TRUTH_PATH, DATA_DIR, OUTPUT_DIR, model_path=MODEL_PATH)
    validator.run_validation()

# python /root/project/ECG/ecg_changgen/code/report/ecg_validation_framework.py --model /root/project/ECG/ecg_changgen/model/clinical_cnn_mitbih_500hz.h5 --ground_truth /root/project/ECG/ecg_changgen/all_report.json --data_dir /root/project/ECG/ecg_changgen/real_data --output_dir /root/project/ECG/ecg_changgen/code/report/validation_results