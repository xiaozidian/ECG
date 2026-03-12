# validation_results - 模型验证结果

## 数据内容
临床 CNN 模型在真实患者数据上的验证输出。

## 目录结构
- `validation_report_clinical_cnn_mitbih_500hz.json` - 验证报告（已上传至 GitHub）
- `processed/` - 10 个患者文件夹，每个包含预测结果 CSV（未上传）

## processed/ 中的数据
按患者 ID 分文件夹，每个包含 `*_predictions.csv`，记录每个心搏的预测分类结果。

## 总大小
约 16 MB

## 如何生成
运行验证脚本，使用 `ecg_changgen/model/` 中的模型对 `ecg_changgen/real_data/` 进行推理。
