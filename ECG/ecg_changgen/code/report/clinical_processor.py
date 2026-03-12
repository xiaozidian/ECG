import numpy as np
import pandas as pd
import neurokit2 as nk
import os
import tensorflow as tf
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D

def get_model():
    nclass = 3
    # 500Hz, window 748
    inp = Input(shape=(748, 1))
    
    # Block 1
    img_1 = Convolution1D(16, kernel_size=15, activation=activations.relu, padding="same")(inp)
    img_1 = Convolution1D(16, kernel_size=15, activation=activations.relu, padding="same")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    
    # Block 2
    img_1 = Convolution1D(32, kernel_size=9, activation=activations.relu, padding="same")(img_1)
    img_1 = Convolution1D(32, kernel_size=9, activation=activations.relu, padding="same")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    
    # Block 3
    img_1 = Convolution1D(64, kernel_size=5, activation=activations.relu, padding="same")(img_1)
    img_1 = Convolution1D(64, kernel_size=5, activation=activations.relu, padding="same")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    
    # Block 4
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax, name="dense_3_mitbih")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    return model

from scipy.stats import kurtosis

def read_data(file_path):
    """
    Reads .DATA file (3-channel, 16-bit, 500Hz).
    Returns all 3 channels.
    """
    # 3 Channels, 500Hz
    num_channels = 3
    fs = 500
    
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    
    data_int16 = np.frombuffer(raw_data, dtype=np.int16)
    
    # Reshape
    # If the file length is not a multiple of num_channels, truncate
    remainder = len(data_int16) % num_channels
    if remainder != 0:
        data_int16 = data_int16[:-remainder]
        
    data = data_int16.reshape(-1, num_channels)
    
    return data, fs

def select_best_channel(data):
    """
    Selects the best channel based on Kurtosis (signal quality).
    ECG signals typically have high kurtosis due to R-peaks.
    """
    best_ch = 0
    max_kurt = -np.inf
    
    # Check first 1 minute or max 100000 points to save time
    limit = min(len(data), 300000) # 10 minutes at 500Hz
    
    print("Evaluating channel quality...")
    for i in range(data.shape[1]):
        sig = data[:limit, i]
        # Calculate Kurtosis
        k = kurtosis(sig)
        print(f"Channel {i} Kurtosis: {k:.2f}")
        
        if k > max_kurt:
            max_kurt = k
            best_ch = i
            
    print(f"Selected Channel {best_ch} as the best signal.")
    return data[:, best_ch]

def filter_twave_misdetection(r_peaks_indices, ecg_signal, sampling_rate, refractory_ms=400):
    r_peaks_indices = np.asarray(r_peaks_indices, dtype=int)
    if len(r_peaks_indices) < 2:
        return r_peaks_indices

    refractory = int(refractory_ms * sampling_rate / 1000.0)
    if refractory <= 1:
        return r_peaks_indices

    kept = [int(r_peaks_indices[0])]
    for p in r_peaks_indices[1:]:
        p = int(p)
        last = kept[-1]
        if p <= last:
            continue

        if p - last < refractory:
            amp_last = float(abs(ecg_signal[last])) if 0 <= last < len(ecg_signal) else 0.0
            amp_p = float(abs(ecg_signal[p])) if 0 <= p < len(ecg_signal) else 0.0

            if amp_last <= 0 and amp_p <= 0:
                continue

            if amp_p > amp_last:
                kept[-1] = p
            continue

        kept.append(p)

    return np.array(kept, dtype=int)

def process_clinical_record(file_path, output_dir='input', model_path='baseline_cnn_mitbih.h5'):
    """
    Processes a clinical .DATA file and generates predictions.csv.
    """
    record_name = os.path.splitext(os.path.basename(file_path))[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_csv_path = os.path.join(output_dir, f'{record_name}_predictions.csv')
    
    print(f"Processing clinical record: {record_name}")
    
    # 1. Read Data
    try:
        data, fs_original = read_data(file_path)
        # Select best channel
        signal = select_best_channel(data)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

    print(f"Signal loaded. Length: {len(signal)}, fs: {fs_original}")
    
    # 2. Clean signal (SKIPPED to match training data)
    print("Cleaning signal (Skipped to match training data)...")
    # ecg_cleaned_high_res = nk.ecg_clean(signal, sampling_rate=fs_original, method="neurokit")
    ecg_cleaned_high_res = signal
    
    # 3. Resample (SKIPPED for 500Hz)
    target_fs = 500
    print(f"Keeping original sampling rate: {target_fs}Hz")
    ecg_cleaned = ecg_cleaned_high_res # nk.signal_resample(ecg_cleaned_high_res, sampling_rate=fs_original, desired_sampling_rate=target_fs)
    
    # 4. R-peaks (on signal)
    print("Detecting R-peaks...")
    try:
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=target_fs)
        r_peaks_indices = rpeaks['ECG_R_Peaks']
    except Exception as e:
        print(f"R-peak detection failed: {e}")
        return None
        
    r_peaks_indices = r_peaks_indices.astype(int)
    before_n = len(r_peaks_indices)
    r_peaks_indices = filter_twave_misdetection(r_peaks_indices, ecg_cleaned, sampling_rate=target_fs)
    after_n = len(r_peaks_indices)
    print(f"Detected {before_n} beats. After T-wave filter: {after_n} beats.")
    
    # 5. Segment
    print("Segmenting...")
    segments = []
    valid_r_peaks = []
    
    # Window: -200 to +548 (Total 748) for 500Hz
    PRE_R = 200
    POST_R = 548
    
    for r_peak in r_peaks_indices:
        start = r_peak - PRE_R
        end = r_peak + POST_R
        if start >= 0 and end < len(ecg_cleaned):
            seg = ecg_cleaned[start:end]
            mean = np.mean(seg)
            var = np.mean((seg - mean) ** 2)
            std = np.sqrt(var) if var > 0 else 1e-6
            seg = (seg - mean) / std
            segments.append(seg)
            valid_r_peaks.append(r_peak)
            
    if not segments:
        print("No valid segments found.")
        return None

    X_data = np.array(segments)
    # Reshape for CNN: (N, 748, 1)
    X_data = X_data[..., np.newaxis]
    
    # 6. Predict
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found.")
        return None
        
    print("Loading model and predicting...")
    try:
        model = get_model()
        model.load_weights(model_path)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None
    predictions = model.predict(X_data, batch_size=1024, verbose=1)
    
    # --- Post-processing Rules ---
    # Apply rules to correct predictions based on RR intervals and probabilities
    
    # Calculate RR intervals in seconds
    rr_intervals = np.diff(r_peaks_indices) / target_fs
    # Insert first RR as mean (or 0)
    rr_intervals = np.insert(rr_intervals, 0, np.mean(rr_intervals))
    
    final_preds = []
    
    # Rolling mean for adaptive RR checking
    window_size = 10
    rr_history = []
    
    for i, (pred_probs, rr) in enumerate(zip(predictions, rr_intervals)):
        pred_class = np.argmax(pred_probs)
        confidence = pred_probs[pred_class]
        
        # Maintain running mean of "Normal" RR intervals
        if len(rr_history) > 0:
            mean_rr = np.mean(rr_history)
        else:
            mean_rr = rr
            
        # --- Rule 1: S (Supraventricular) correction ---
        # STRICT RULE: S beats MUST be premature.
        # MIT-BIH models often over-predict S on continuous data.
        if pred_class == 1: # S
            # Definition of prematurity: RR < 95% of running mean (Relaxed from 90%)
            is_premature = rr < 0.95 * mean_rr
            
            if not is_premature:
                # If not premature, it is almost certainly NOT an S beat (S beats are APCs)
                # Force to N regardless of confidence (unless extremely high, but even then doubtful in this context)
                pred_class = 0 
            else:
                # Even if premature, it might be noise or T-wave detection.
                # Relaxed confidence for S (from 0.90 to 0.60) to improve recall
                if confidence < 0.60:
                    pred_class = 0
        
        # --- Rule 2: V (Ventricular) correction ---
        # V beats are broad and distinct.
        # Reduce False Positives by requiring high confidence.
        if pred_class == 2: # V
            # Relaxed threshold (from 0.95 to 0.75) to improve recall
            if confidence < 0.75: 
                pred_class = 0
                
        # --- Rule 3: Artifact/Noise Rejection ---
        # Extremely short RR intervals (< 300ms, i.e., > 200 bpm) are often noise
        # unless part of a known Tachycardia run, but singletons are likely noise.
        if rr < 0.3:
            # If the beat is classified as S or V but is physically impossible/unlikely, revert to N (or Q)
            # This helps avoid counting artifacts as arrhythmias
            pred_class = 0
                
        # Update history if it's a Normal beat (or was converted to N)
        if pred_class == 0:
            rr_history.append(rr)
            if len(rr_history) > window_size:
                rr_history.pop(0)
                
        final_preds.append(pred_class)
        
    pred_classes = np.array(final_preds)
    
    # 7. Save
    class_names = {0: 'N', 1: 'S', 2: 'V'}
    
    df_res = pd.DataFrame({
        'Beat_Index': range(len(pred_classes)),
        'R_Peak_Pos': valid_r_peaks,
        'Predicted_Class': pred_classes,
        'Class_Name': [class_names.get(c, 'Unknown') for c in pred_classes]
    })
    
    df_res.to_csv(output_csv_path, index=False)
    print(f"Saved predictions to {output_csv_path}")
    return output_csv_path
