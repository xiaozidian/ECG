# PTB-XL - 公开 12 导联心电数据集

## 数据来源
PTB-XL 是 PhysioNet 上的公开大规模 12 导联 ECG 数据集，包含 21,799 条 10 秒心电记录。

## 原始目录结构
| 文件/目录 | 大小 | 说明 |
|---|---|---|
| `ptbxl_records500/` | 2.6 GB | 500Hz 采样的 WFDB 格式记录（.dat + .hea），按编号分 22 个子文件夹 |
| `ptbxl_records500.zip` | 1.4 GB | 上述目录的压缩包 |
| `ptbxl_database.csv` | 6.3 MB | 主元数据/标签 CSV（已上传至 GitHub） |
| `scp_statements.csv` | 9.5 KB | SCP 诊断语句定义（已上传至 GitHub） |
| `LICENSE.txt` | 14 KB | 数据集许可证（已上传至 GitHub） |

## 总大小
约 4 GB

## 如何获取
公开数据集，可从 PhysioNet 免费下载：
https://physionet.org/content/ptb-xl/

```bash
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```

也可使用项目根目录的 `hf_download.py` 从 HuggingFace 下载。
