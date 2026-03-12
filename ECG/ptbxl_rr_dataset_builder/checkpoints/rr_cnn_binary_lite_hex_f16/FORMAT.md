# rr_cnn_binary_lite：FP16 权重文本格式约定

本目录包含将 checkpoint `/root/project/ECG/ptbxl_rr_dataset_builder/checkpoints/rr_cnn_binary_lite.pt` 的 `state_dict` 导出的权重文本，便于硬件仿真加载。

导出目录（本文件所在目录）：

- `/root/project/ECG/ptbxl_rr_dataset_builder/checkpoints/rr_cnn_binary_lite_hex_f16`

配套清单：

- `manifest.json`：记录每个张量对应的文件名、shape、numel、格式信息

## 1. 文件与张量对应关系

- `conv1.weight.hex` ⇔ `conv1.weight`
- `conv2.weight.hex` ⇔ `conv2.weight`
- `conv3.weight.hex` ⇔ `conv3.weight`
- `conv4.weight.hex` ⇔ `conv4.weight`
- `fc1.weight.hex` ⇔ `fc1.weight`
- `fc1.bias.hex` ⇔ `fc1.bias`
- `fc2.weight.hex` ⇔ `fc2.weight`
- `fc2.bias.hex` ⇔ `fc2.bias`

张量的 **shape** 以 `manifest.json` 为准（不同训练参数/模型结构可能导致 shape 变化）。

## 2. 数值编码（FP16）

- 每个参数存为 **IEEE754 float16（binary16）**
- 每个参数占 **16 bit = 2 字节**
- 文本表示为 **十六进制字符串**

### 2.1 每行一个参数（one word per line）

- 每个文件按行存储参数
- **一行 = 一个 16-bit word**
- 一行固定 **4 个十六进制字符**（对应 2 字节）

### 2.2 字节序（endianness）

采用 **小端序（little-endian）** 写入文本：

- 一行的 hex 字符串表示的是“该 16-bit word 的 2 个字节”的小端顺序
- 若一行内容为 `a1b2`，则对应两字节为：`0xA1 0xB2`
- 作为 16-bit 无符号整数 `uint16` 解读时，该 word 的数值为：`0xB2A1`
- 再按 IEEE754 float16 解释该 `uint16` 的 bit pattern，即得到参数值

## 3. 张量展平与排列顺序

- 每个张量导出前会先转为 `float16`
- 然后按 **C-order（行优先）** 展平为一维序列写入（等价于 NumPy `reshape(-1)` 默认顺序）

这对硬件侧重建非常关键：

1. 按文件顺序逐行读取得到一维 `float16` 序列
2. 依据 `manifest.json` 中的 `shape`，按 C-order 还原为多维张量

## 4. 读取与还原示例

### 4.1 Python 示例（读取一个 .hex 还原为 float16 数组）

```python
import numpy as np
import struct

def load_hex_f16(path: str) -> np.ndarray:
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            h = line.strip()
            b = bytes.fromhex(h)           # 2 bytes (little-endian)
            u16 = struct.unpack("<H", b)[0]
            v = np.array([u16], dtype=np.uint16).view(np.float16)[0]
            vals.append(v)
    return np.asarray(vals, dtype=np.float16)
```

### 4.2 C/C++ 思路（伪代码）

```c
// 每行读入4个hex字符 -> 2字节 little-endian
uint8_t lo = ...; // 第1个字节
uint8_t hi = ...; // 第2个字节
uint16_t bits = ((uint16_t)hi << 8) | lo;

// bits 即 float16 的 bit pattern
// 若硬件本身支持 FP16，可直接将 bits 解释为 FP16
// 若不支持，可做 float16->float32 的软件转换用于验证
```

## 5. 校验口径

该目录的文本表示与如下流程严格一致：

1. 从 checkpoint 取出张量（float32）
2. 执行 `float32 -> float16` 量化（NumPy/PyTorch 标准转换）
3. 将量化后的 float16 **bit pattern** 以 little-endian 方式写成 hex 文本（每行一个参数）

因此，若你在硬件仿真侧按本格式读取并解释为 float16，结果应与上述量化结果逐元素一致。
