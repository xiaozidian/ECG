# output_ptbxl - PTB-XL RR 间期处理后数据

## 数据内容
从 PTB-XL 数据集提取的 RR 间期特征样本，以 `.npz` 格式存储。

## 文件格式
- 共 12,858 个 `.npz` 文件，按原始记录编号命名
- 每个文件包含提取的 RR 间期特征和对应标签

## 总大小
约 53 MB

## 如何生成
运行 `build_ptbxl_rr_dataset.py`，需要先准备好 PTB-XL 原始数据集。
