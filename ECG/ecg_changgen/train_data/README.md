# train_data - CNN 训练数据集（CSV 格式）

## 数据内容
用于心搏分类 CNN 模型训练和测试的 CSV 数据集。

## 文件列表
| 文件名 | 大小 | 说明 |
|---|---|---|
| `clinical_mitbih_train.csv` | 1.5 GB | 临床 + MIT-BIH 混合训练集 |
| `clinical_mitbih_test.csv` | 335 MB | 临床 + MIT-BIH 混合测试集 |
| `mitbih_train.csv` | 392 MB | MIT-BIH 心律失常数据库训练集 |
| `mitbih_test.csv` | 98 MB | MIT-BIH 心律失常数据库测试集 |
| `ptbdb_abnormal.csv` | 47 MB | PTB 诊断数据库 - 异常样本 |
| `ptbdb_normal.csv` | 18 MB | PTB 诊断数据库 - 正常样本 |

## 总大小
约 2.4 GB

## 如何获取
- MIT-BIH / PTB 数据：可从 PhysioNet 下载原始数据后用脚本处理生成
- Clinical 数据：由 `code/generate_beat_data.py` 从 `real_data/` 和 `DMdata/` 提取生成
- 也可使用项目根目录的 `hf_download.py` / `hf_download.sh` 下载
