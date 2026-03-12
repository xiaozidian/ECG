#!/usr/bin/env bash
set -e

# ================================
# Hugging Face 通用下载脚本 (Bash版)
# ================================

# -------- 参数 --------
REPO_ID="$1"          # e.g. meta-llama/Llama-2-7b-hf
TARGET_DIR="$2"       # 可选：下载目录（为空则用默认缓存）
MAX_RETRIES="${3:-5}" # 可选：最大重试次数，默认 5

# -------- 检查参数 --------
if [ -z "$REPO_ID" ]; then
  echo "Usage:"
  echo "  $0 <repo_id> [target_dir] [max_retries]"
  echo
  echo "Example:"
  echo "  $0 meta-llama/Llama-2-7b-hf"
  echo "  $0 meta-llama/Llama-2-7b-hf ./models/llama2 10"
  exit 1
fi

# -------- 基础环境 --------
# 禁用遥测
export HF_HUB_DISABLE_TELEMETRY=1
# 强制关闭 hf_transfer（虽然速度快但有时不稳定，设为0更稳；如果追求速度可设为1并安装hf_transfer）
export HF_HUB_ENABLE_HF_TRANSFER=0

# -------- 命令构造 --------
# 基础命令
CMD=(huggingface-cli download "$REPO_ID")

# 如果指定了目标目录
if [ -n "$TARGET_DIR" ]; then
  mkdir -p "$TARGET_DIR"
  CMD+=("--local-dir" "$TARGET_DIR")
  # 注意：新版 huggingface-cli 默认 --local-dir-use-symlinks False，即直接下载文件
  # 如果你希望使用 symlinks 连接到缓存，可以加上 --local-dir-use-symlinks True
else
  # 未指定目录，下载到默认缓存
  echo "Target: Default Cache"
fi

# 启用断点续传
CMD+=("--resume-download")

# -------- 重试逻辑 --------
ATTEMPT=1
while [ $ATTEMPT -le $MAX_RETRIES ]; do
  echo "=============================="
  echo "Attempt $ATTEMPT / $MAX_RETRIES"
  echo "Command: ${CMD[*]}"
  echo "=============================="

  # 执行命令
  set +e # 临时允许失败以便捕获
  "${CMD[@]}"
  EXIT_CODE=$?
  set -e # 恢复

  if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Download completed successfully."
    exit 0
  else
    echo "⚠️  Download failed (code $EXIT_CODE). Retrying in 10 seconds..."
    sleep 10
  fi

  ATTEMPT=$((ATTEMPT + 1))
done

echo "❌ Download failed after $MAX_RETRIES attempts."
exit 2
