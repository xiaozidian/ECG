#!/bin/bash
# 改进的训练配置 - 缓解当前数据集问题

# 方案1: 降低重采样强度，更接近真实分布
python /root/project/ECG/ecg_changgen/code/clinical_cnn_train.py \
  --train-data-dir /root/project/ECG/ecg_changgen/train_hospital \
  --epochs 30 \
  --sampling gpu_resample \
  --resample-weights 0.96,0.02,0.02 \
  --min-class-fraction 0.005 \
  --max-class-fraction 0.05 \
  --loss focal \
  --focal-gamma 3.0 \
  --focal-alpha auto \
  --focal-alpha-power 0.8 \
  --s-threshold 0.75 \
  --v-threshold 0.80 \
  --model-save-path /root/project/ECG/ecg_changgen/model/clinical_cnn_improved_v1.h5

echo "=============================="
echo "方案1完成"
echo "=============================="

# 方案2: 使用自动权重
python /root/project/ECG/ecg_changgen/code/clinical_cnn_train.py \
  --train-data-dir /root/project/ECG/ecg_changgen/train_hospital \
  --epochs 30 \
  --sampling gpu_resample \
  --resample-weights auto \
  --resample-alpha 0.3 \
  --loss focal \
  --focal-gamma 4.0 \
  --focal-alpha auto \
  --s-threshold 0.80 \
  --v-threshold 0.85 \
  --model-save-path /root/project/ECG/ecg_changgen/model/clinical_cnn_improved_v2.h5

echo "=============================="
echo "方案2完成"
echo "=============================="

# 方案3: 不使用重采样，只用focal loss
python /root/project/ECG/ecg_changgen/code/clinical_cnn_train.py \
  --train-data-dir /root/project/ECG/ecg_changgen/train_hospital \
  --epochs 30 \
  --sampling none \
  --loss focal \
  --focal-gamma 5.0 \
  --focal-alpha auto \
  --focal-alpha-power 1.0 \
  --s-threshold 0.85 \
  --v-threshold 0.90 \
  --use-class-weight \
  --model-save-path /root/project/ECG/ecg_changgen/model/clinical_cnn_improved_v3.h5

echo "=============================="
echo "三个方案全部完成"
echo "预期效果:"
echo "  - Precision会显著提升(可能到30-50%)"
echo "  - Recall会下降(但这是正常的权衡)"
echo "  - F1-score应该会提升"
echo "=============================="
