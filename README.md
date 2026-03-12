# ECG - 心电图智能分析系统

基于深度学习的心电图（ECG）自动分析系统，支持心搏分类、心律失常检测、HRV 分析及临床报告生成。项目包含从原始心电信号处理到临床报告输出的完整流水线，以及面向嵌入式硬件部署的轻量模型。

## 项目结构

```
.
├── ECG/
│   ├── ecg_changgen/          # 核心项目：临床心电分析系统
│   │   ├── code/              # 全部源代码
│   │   ├── model/             # 训练好的模型权重 (.h5)
│   │   ├── DMdata/            # [数据集] Holter 原始数据 (~13GB)
│   │   ├── real_data/         # [数据集] 医院患者数据 (~5.9GB)
│   │   ├── train_data/        # [数据集] 训练用 CSV (~2.4GB)
│   │   ├── train_hospital/    # [数据集] 训练用 numpy (~5.6GB)
│   │   ├── validation_results/    # 验证报告
│   │   ├── validation_results_v3/ # V3 验证报告
│   │   └── report_figures/    # 分析图表
│   │
│   ├── ptbxl_rr_dataset_builder/  # RR 间期轻量模型（面向硬件部署）
│   │   ├── *.py               # 数据构建与训练代码
│   │   ├── checkpoints/       # 模型权重 (.pt + hex 导出)
│   │   └── output_*/          # [数据集] 处理后的数据
│   │
│   ├── PTB-XL/                # [数据集] PTB-XL 公开数据集元数据
│   ├── torch_ecg/             # 开源 ECG 工具库 (submodule)
│   └── validation_results/    # 顶层验证结果
│
├── hf_download.py             # HuggingFace 数据集下载脚本
├── hf_download.sh             # 下载辅助脚本
└── train.log                  # 训练日志
```

> 标注 `[数据集]` 的文件夹未上传至 GitHub，每个文件夹内有 `README.md` 说明数据来源和获取方式。

## 核心功能

### 1. 心搏分类 (Beat Classification)

基于 1D-CNN 的心搏三分类模型：
- **N** - 正常心搏
- **S** - 室上性早搏 (PAC)
- **V** - 室性早搏 (PVC)

训练数据融合了 MIT-BIH 心律失常数据库和医院临床数据，支持 GPU 加速训练、类别不平衡重采样、数据增强、混合精度训练等特性。

**关键代码：** `ECG/ecg_changgen/code/clinical_cnn_train.py`

### 2. 临床报告生成

从原始 .DATA 二进制文件到完整临床报告的自动化流水线：
- 多导联信号读取与自动信道选择（基于峰度评估）
- R 波检测与 T 波误检校正
- 心率统计（平均/最慢/最快，10 秒滑动窗口平滑）
- 心律失常事件计数与 RR 间期后处理校正
- HRV 时域/频域分析（5 分钟分段 Welch 功率谱）

**关键代码：** `ECG/ecg_changgen/code/report/`

### 3. RR 间期轻量模型（硬件部署）

面向嵌入式/FPGA 的超轻量 CNN，仅使用 RR 间期特征进行心律分类：
- 节律分类（SR / AFIB / STACH / SBRAD / SARRH / OTHER）
- 异位搏动检测（二分类）
- 模型权重导出为 float16 hex 格式，可直接烧录硬件

**关键代码：** `ECG/ptbxl_rr_dataset_builder/`

## 模型

### 心搏分类 CNN

| 模型文件 | 说明 |
|---|---|
| `baseline_cnn.h5` | MIT-BIH 基线模型 |
| `my_cnn_v1.h5` | 自定义 CNN v1 |
| `clinical_cnn_mitbih_500hz.h5` | 临床+MIT-BIH 混合训练，500Hz |
| `clinical_cnn_improved_v1.h5` | 改进版 v1 |
| `clinical_cnn_improved_v2.h5` | 改进版 v2 |

### RR 间期轻量 CNN

| 模型文件 | 说明 |
|---|---|
| `rr_cnn.pt` | 完整 RR CNN（多分类） |
| `rr_cnn_binary_lite.pt` | 轻量二分类 |
| `rr_cnn_binary_lite_2conv.pt` | 超轻量 2-conv 架构（适合 FPGA） |

## 算法改进记录

项目经历了 9 轮重要的算法迭代优化（详见 `ECG/ecg_changgen/改进记录.md`）：

1. **带通滤波** - Butterworth 滤波器消除基线漂移和高频干扰
2. **T 波误检校正** - 基于不应期和波峰幅度比较的后处理
3. **多导联自动选择** - 基于峰度 (Kurtosis) 的信号质量评估
4. **早搏误报抑制** - 结合 RR 间期节律特征的 S/V 校正规则
5. **极值心率优化** - 4 搏滑动平均 → 10 秒窗口平滑
6. **频域 HRV 修正** - 单位换算 (s²→ms²) + 5 分钟分段 Welch 估计
7. **最快心率修正** - T 波误检剔除 + 自适应窗口
8. **最长 RR 间期** - 与心率统计解耦，独立宽松范围筛选
9. **网格搜索优化** - 10 万+样本拟合最优参数（MAE 7.55 bpm）

## 数据集

数据集未上传到 GitHub，各文件夹内均有 README 说明获取方式。

| 数据集 | 大小 | 来源 | 获取方式 |
|---|---|---|---|
| DMdata | 13 GB | 医院 Holter 记录仪 | 联系数据提供方 |
| real_data | 5.9 GB | 医院患者心电数据 | 联系数据提供方 |
| train_data | 2.4 GB | MIT-BIH + 临床混合 | `generate_beat_data.py` 生成 |
| train_hospital | 5.6 GB | 临床数据预处理 | `generate_beat_data.py` 生成 |
| PTB-XL | 4 GB | PhysioNet 公开数据集 | [PhysioNet](https://physionet.org/content/ptb-xl/) |
| output_ptbxl* | 236 MB | PTB-XL 处理后 | `build_ptbxl_rr_dataset.py` 生成 |

## 依赖

- Python 3.8+
- TensorFlow / Keras（心搏分类 CNN）
- PyTorch（RR 间期轻量模型）
- NumPy, SciPy, NeuroKit2（信号处理）
- wfdb（WFDB 格式读取）

## 快速开始

```bash
# 克隆仓库（含 submodule）
git clone --recurse-submodules https://github.com/xiaozidian/ECG.git

# 下载 PTB-XL 数据集
python hf_download.py

# 构建 RR 间期数据集
cd ECG/ptbxl_rr_dataset_builder
python build_ptbxl_rr_dataset.py

# 训练 RR CNN
python train_rr_cnn_binary_lite.py

# 训练心搏分类 CNN（需要先准备训练数据）
cd ../ecg_changgen
python code/clinical_cnn_train.py --epochs 30

# 生成临床报告
python code/report/batch_generate_reports.py
```
