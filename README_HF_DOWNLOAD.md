# Hugging Face 通用下载脚本工具集

本仓库包含两个通用的 Hugging Face 资源下载脚本（Python 版和 Bash 版），旨在解决网络不稳、大文件下载中断等问题。

## 功能特性

- **断点续传**：自动检测已下载的部分，只下载剩余内容。
- **自动重试**：遇到网络错误自动等待并重试，支持自定义重试次数。
- **灵活的存储位置**：
  - 支持下载到 Hugging Face 默认缓存（便于统一管理）。
  - 支持下载到指定目录（便于离线部署或特定项目使用）。
- **多类型支持**：支持 Model（模型）、Dataset（数据集）和 Space（空间）。
- **兼容性提示**：下载完成后，自动生成 Python 加载代码（`from_pretrained`）。

---

## 1. Python 脚本 (`hf_download.py`)

Python 版本功能最全，支持细粒度的文件过滤和参数控制。

### 依赖安装
```bash
pip install huggingface_hub
```

### 基本用法

**下载模型到默认缓存：**
```bash
python hf_download.py meta-llama/Llama-2-7b-hf
```

**下载到指定目录（推荐）：**
```bash
python hf_download.py meta-llama/Llama-2-7b-hf --dir ./models/llama2
```

### 高级参数

| 参数 | 说明 | 示例 |
| :--- | :--- | :--- |
| `repo_id` | **(必填)** 仓库ID | `google-bert/bert-base-uncased` |
| `--dir` | 本地存储目录 | `--dir ./my_model` |
| `--type` | 仓库类型 (model, dataset, space) | `--type dataset` |
| `--token` | HF Token (私有模型必填) | `--token hf_xxx` |
| `--retries` | 最大重试次数 (默认 10) | `--retries 20` |
| `--allow` | 只下载匹配的文件模式 | `--allow "*.json" "*.bin"` |
| `--ignore` | 忽略匹配的文件模式 | `--ignore "*.msgpack"` |

**只下载特定文件示例：**
```bash
python hf_download.py google-bert/bert-base-uncased --allow "*.json" "*.safetensors"
```

---

## 2. Bash 脚本 (`hf_download.sh`)

Bash 版本依赖 `huggingface-cli`，适合快速在服务器环境使用，无需编写 Python 代码。

### 依赖安装
确保安装了 `huggingface_hub`（会自动安装 CLI 工具）：
```bash
pip install huggingface_hub
```

### 使用方法

首先赋予执行权限：
```bash
chmod +x hf_download.sh
```

**语法：**
```bash
./hf_download.sh <repo_id> [target_dir] [max_retries]
```

**示例：**

1.  **简单下载（默认缓存）**：
    ```bash
    ./hf_download.sh meta-llama/Llama-2-7b-hf
    ```

2.  **下载到指定目录**：
    ```bash
    ./hf_download.sh meta-llama/Llama-2-7b-hf ./models/llama2
    ```

3.  **指定重试次数**：
    ```bash
    ./hf_download.sh meta-llama/Llama-2-7b-hf ./models/llama2 20
    ```

---

## 常见问题与技巧

### 关于 `from_pretrained` 加载

下载完成后，脚本会提示你如何加载模型。

- **如果你下载到了默认缓存**：
  直接使用 Repo ID 加载：
  ```python
  model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf")
  ```

- **如果你下载到了指定目录**（例如 `./models/llama2`）：
  使用本地路径加载：
  ```python
  model = AutoModel.from_pretrained("./models/llama2")
  ```

### 环境变量说明
脚本内部已自动设置以下环境变量以优化稳定性：
- `HF_HUB_DISABLE_TELEMETRY=1`: 禁用遥测。
- `HF_HUB_ENABLE_HF_TRANSFER=0`: 禁用 `hf_transfer` 加速（该库虽快但对网络要求高，易报错，关闭后使用默认下载器更稳健）。

如需修改默认缓存位置，请在运行脚本前设置：
```bash
export HF_HOME="/your/new/cache/path"
```
