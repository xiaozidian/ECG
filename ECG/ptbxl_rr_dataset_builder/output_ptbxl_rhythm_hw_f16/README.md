# output_ptbxl_rhythm_hw_f16 - 硬件部署用 hex 数据（float16）

## 数据内容
将心律分类数据集导出为硬件友好的 hex 格式（float16 精度），用于嵌入式/FPGA 部署。

## 文件列表
- `x_ch0_f16_lines.hex` - 通道0特征
- `x_ch1_f16_lines.hex` - 通道1特征
- `hrv_f16_lines.hex` - HRV 特征
- `y_u16.hex` - 标签
- `manifest.json` - 数据清单

## 总大小
约 4.7 MB

## 如何生成
运行 `export_hw_dataset_lines.py`。
