# train_rr_cnn.py 模型优化汇报

## 当前模型结构（train_rr_cnn.py）

### 1) 输入与特征构成
- 任务：RR-only 的 6 类节律分类（SR / AFIB / STACH / SBRAD / SARRH / OTHER_RHYTHM）
- 每条样本从 npz 中加载 rr 序列（默认 rr_key=rr_ms，若 npz 内存在 rr_key 则优先使用）
- RR 处理流程：
  - 生理范围裁剪：300–2000 ms
  - rr_ms 做全局 z-score（使用训练集统计量 rr_mean / rr_std）
  - delta_rr = diff(rr_ms, prepend=rr_ms[0]) 做对齐后再 z-score（使用训练集统计量 delta_mean / delta_std）
  - padding/trim 到 seq_len（默认 128）：
    - rr_z padding 用 rr_z[-1]（延续末端水平）
    - delta_z padding 用 0（避免人为“持续变化”假信号）
- 双通道序列输入 x：
  - 形状：x ∈ R^{B×2×T}，其中 T=seq_len
  - ch0=rr_z，ch1=delta_z
- HRV 手工特征 hrv（4 维）：
  - 指标：SDNN / RMSSD / pNN50 / Sample Entropy
  - 使用训练集统计量做 z-score（hrv_mean / hrv_std）
  - 形状：hrv ∈ R^{B×4}

### 2) 模型分支与可选架构
脚本提供两种可选主干（--model）：

#### A) RRResNetLite（--model resnet）
- 输入：x ∈ R^{B×2×T}，hrv ∈ R^{B×4}
- 主干：多层 1D 卷积堆叠（stride=1），不使用 MaxPool，避免时间细节丢失
- 聚合：对时间维做 mean pooling 得到序列表征 f ∈ R^{B×C}
- 融合：concat([f, hrv]) → R^{B×(C+4)}
- 分类头：线性层输出 logits ∈ R^{B×6}

#### B) RRTCNAttn（--model tcn_attn，默认）
- 输入：x ∈ R^{B×2×T}，hrv ∈ R^{B×4}
- TCN：多层 TemporalBlock，dilation 按层指数增长（2^idx），扩大感受野以捕捉长程依赖/周期性
- Attention：TemporalAttention 通过 1×1 conv 生成时间权重，对异常点/突变点自适应加权汇聚
- 融合：concat([attn_pool(x), hrv]) → 分类头输出 logits ∈ R^{B×6}

### 3) 训练目标与不平衡处理
- 损失函数可选：
  - CrossEntropy（默认）
  - FocalLoss（--loss focal，支持 focal_gamma）
- 类别不平衡：按训练集类别频次计算 class weights，并注入到 CE/Focal 中

### 4) 训练/验证/测试数据流（接口约定）
- Dataset 返回：(x, hrv, y)
- Model forward 约定：logits = model(x, hrv)
- 评估输出：
  - confusion matrix
  - per-class precision/recall/F1 与 macro-F1

## 背景与目标
- 现有 RR-only 节律分类模型在宏平均 F1 与小类（如 SARRH）上表现较弱
- 目标是以最小改动提升分类可分性与稳定性，同时保持训练流程可复现

## 已落实的优化点

### 1) 归一化修正：保留 RR 绝对尺度
- 原先 min-max 归一化会削弱绝对快慢信息
- 现改为：先做生理范围裁剪（300–2000 ms），再用训练集均值/方差做全局 z-score
- 影响：增强 SBRAD/STACH 等“快慢”类区分度，减少 SR 被过度归一化导致的误判

### 2) 双通道输入：RR + ΔRR
- ch0：rr_ms（z-score）
- ch1：delta_rr（beat-to-beat 变化）
- 影响：增强 AFIB/SARRH 等“波动型”节律的可分性

### 3) 序列长度调整：seq_len=128
- 在归一化修正后，提升窗口长度以稳定统计特征
- 默认 seq_len 已设置为 128

### 4) ΔRR 对齐与 padding 修正
- 对齐方式：delta_rr = diff(rr_ms, prepend=rr_ms[0])，首位补 0 保证与 rr 对齐
- padding 方式：对 delta_rr 采用 0 padding，避免复制最后一个差分值造成“持续变化”的假信号

### 5) rr_key 读取逻辑修正
- 优先使用传入的 rr_key，只有当 rr_key 不存在时才回退到 rr_ms
- 避免 rr_key 形同虚设

### 6) 模型结构：RRResNetLite 去除 MaxPool
- 原先大量 MaxPool1d 会丢失精确时间间隔信息
- 改为 stride=1 的卷积堆叠，保留细粒度时间细节

### 7) 新模型：TCN + Attention
- 引入 TCN（膨胀卷积）说明长程依赖
- 引入注意力，对异常点和突然变化更敏感
- 可通过 --model tcn_attn 切换

### 8) HRV 特征融合
在全连接层前拼接 HRV 统计特征：
- SDNN
- RMSSD
- pNN50
- Sample Entropy

HRV 特征使用训练集统计量进行 z-score，并与序列特征拼接后分类

## 训练与评估流程更新
- 数据集输出为 (x, hrv, y)，模型前向改为 (x, hrv)
- 训练、验证、测试全流程已对接 HRV 特征
- checkpoint 保存 hrv_mean/hrv_std 以便推理复现

## 关键实现位置
- train_rr_cnn.py：数据处理、模型结构、训练流程全部集成
- 可通过参数切换模型、窗口长度、损失函数等

## 预期收益
- SR 与 AFIB 混淆减少
- SBRAD / STACH 对快慢信息更敏感
- SARRH / AFIB 对波动与非线性特征更敏感
- 模型对节律的长期依赖关系捕捉更充分

## 最新效果与误差分析（RR-only，6 类节律）

### 1) 总体结论
- 当前结果在 “RR-only（只用 RR 间期序列）做 6 类节律分类” 设定下已达到较强水平
- macro-F1 从最初约 0.38 提升到 0.576，核心类（SR / AFIB / STACH）F1 达到 0.73–0.80+
- 结论：保留绝对尺度的 z-score + delta 通道 + 更长窗口 + 处理细节 的优化路线有效

### 2) 指标概览
- macro-F1：0.576
- per-class F1：
  - SR：0.875
  - AFIB：0.731
  - STACH：0.795
  - SBRAD：0.420
  - SARRH：0.367
  - OTHER：0.269

直观含义：
- “有明确 RR 特征”的类（SR/AFIB/STACH）已接近 RR-only 信息上限
- 当前主要短板为 OTHER（混合桶）与 SARRH（异质类），其次为 SBRAD

### 3) 混淆矩阵解读
类别顺序：[SR, AFIB, STACH, SBRAD, SARRH, OTHER]

confusion matrix：
```
SR    [1308  37  21  90 120  67]  total=1643
AFIB  [  3 137   1   3  14  12]  total=170
STACH [  0   4  68   0   2   2]  total=76
SBRAD [ 10   1   0  43   1   5]  total=60
SARRH [  5  11   1   3  47   1]  total=68
OTHER [ 20  15   4   6   4  25]  total=74
```

逐类要点：
- SR（真 1643）：主要误判到 SARRH / SBRAD / OTHER
  - SR→SARRH：120（轻度不齐 SR 容易被吸到 SARRH）
  - SR→SBRAD：90（慢心率 SR 与 SBRAD 边界本就模糊）
  - SR→OTHER：67（噪声/异常段被吸入混合桶）
- AFIB（真 170）：recall≈80.6%，主要误判到 SARRH 与 OTHER（不齐类型在 RR-only 下容易混）
- STACH（真 76）：recall≈89.5%，非常稳定，说明“绝对尺度 + 变化信息”有效
- SBRAD（真 60）：主要与 SR 互混，属于“慢 SR”导致的标签边界问题
- SARRH（真 68）：recall≈69.1%，但 SR→SARRH 假阳性仍多，precision 偏低导致 F1 不高
- OTHER（真 74）：recall≈33.8%，该类是混合桶，RR-only 下很难兼顾高 precision 与高 recall

### 4) 主要混淆对与提升空间
当前最大“损失来源”：
1. SR ↔ SARRH（SR→SARRH 120）
2. SR ↔ SBRAD（SR→SBRAD 90）
3. OTHER 低 recall + 低 precision（混合桶分散误判）

其中 1) 最关键：降低 “SR 被误判为 SARRH” 将同时改善 SR 与 SARRH 的 precision，从而进一步推高 macro-F1。

### 5) 当前瓶颈（为什么进入天花板区）
- 类别可分性上限（信息论限制）：
  - SARRH 不是单一机制，包含多种“非典型窦律/窦性不齐/间歇性不齐”的混合
  - OTHER 是“其余节律”的混合桶，RR-only 缺少形态信息（P 波、QRS、房扑/早搏形态、导联信息等）
- SR 内部异质性：
  - 慢 SR 与 SBRAD 的边界天然模糊
  - 波动较大的 SR 与 SARRH/AFIB 的边界也不完全清晰

