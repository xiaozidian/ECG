# 1 节律数据集与早搏/异位搏动二分类数据集说明
**已完成的数据集构建**
- 节律主任务数据集已生成：/root/project/ECG/ptbxl_rr_dataset_builder/output_ptbxl_rhythm  
  - 总样本 21064；类别分布：SR 16720，AFIB 1514，STACH 805，SBRAD 631，SARRH 767，OTHER_RHYTHM 627  
  - 汇总见 [summary.json](file:///root/project/ECG/ptbxl_rr_dataset_builder/output_ptbxl_rhythm/summary.json)
- 早搏/异位搏动二分类数据集已生成：/root/project/ECG/ptbxl_rr_dataset_builder/output_ptbxl_ectopy  
  - 总样本 21835；类别分布：NO_ECTOPY 20635，ECTOPY 1200  
  - 汇总见 [summary.json](file:///root/project/ECG/ptbxl_rr_dataset_builder/output_ptbxl_ectopy/summary.json)

**构建方式与可复用命令**
- 脚本已支持一键生成两套新数据集，入口在 [build_ptbxl_rr_dataset.py](file:///root/project/ECG/ptbxl_rr_dataset_builder/build_ptbxl_rr_dataset.py)
- 我执行的命令：  
  ```bash
  python /root/project/ECG/ptbxl_rr_dataset_builder/build_ptbxl_rr_dataset.py --task all --ectopy_include_bigu_trigu
  ```

**标签策略已落地**
- 主任务标签集合：SR / AFIB / STACH / SBRAD / SARRH / OTHER_RHYTHM  
- 副任务 ECTOPY：使用 PVC + BIGU/TRIGU（你要求的可选项已启用）

# 2 多标签->单标签解释
          
多标签变成现在这种单标签，是在构建节律数据集时做了一个“**单标签折叠规则（priority rule）**”：对每条记录的 `scp_codes`（本质是一个 code 集合）只输出**一个**节律类别。

**1) 原始为什么是“多标签”**
- PTB‑XL 的 `scp_codes` 是一个字典（code→置信度/权重），同一条记录里可能同时出现多个 rhythm code，比如：
  - `('AFIB','AFLT')`、`('BIGU','SR')`、`('PSVT','SVTAC')` 等
- 但这种“多节律共现”在 PTB‑XL 里比例很低：在所有含 rhythm 的记录中，**多节律只有 143 条，占 0.7%**。

**2) 折叠为单标签的具体规则是什么**
构建脚本里用 `_label_rhythm()` 把一条记录映射成单标签，逻辑见 [build_ptbxl_rr_dataset.py](file:///root/project/ECG/ptbxl_rr_dataset_builder/build_ptbxl_rr_dataset.py#L69-L86)：

优先级（从高到低）是：
1. 只要含 `AFIB` → 直接标为 **AFIB**
2. 含任意 `{"AFLT","SVARR","SVTAC","PSVT","PACE","BIGU","TRIGU"}` → 标为 **OTHER_RHYTHM**
3. 含 `STACH` → **STACH**
4. 含 `SBRAD` → **SBRAD**
5. 含 `SARRH` → **SARRH**
6. 含 `SR` → **SR**
7. 都不满足 → 丢弃（label=None）

这就是“多标签 → 单标签”的核心：**只取一个最优先的类别**，并把稀少/难分/杂类节律统一进 OTHER。

**3) 为什么最终统计会变成 summary.json 里的那组数**
- 单节律样本直接落到对应类（比如只有 SR 的就算 SR）。
- 多节律样本按上面优先级被“抢占”到某一类：
  - 多节律 143 条里，最终分配为：**OTHER_RHYTHM 108、AFIB 30、SBRAD 3、SARRH 2**
  - 典型例子：
    - `('BIGU','SR')` → OTHER（因为 BIGU 属于 other_codes，优先于 SR）
    - `('AFIB','AFLT')` → AFIB（因为 AFIB 优先级最高）
    - `('SARRH','SBRAD')` → SBRAD（SBRAD 在 SARRH 之前）
- 你看到的类别数（SR 16720, AFIB 1514, STACH 805, SBRAD 631, SARRH 767, OTHER 627）基本等于“该 code 在 PTB‑XL 中的出现数”减去少量被 RR 抽取失败/文件缺失的样本（这里 skipped_short_rr=2）。

**4) 这样折叠有什么影响？**
- 好处：把任务变成标准单标签多分类，训练更简单稳定；并且把稀少/混淆大的节律合并成 OTHER，避免小类学不动。
- 代价：那 0.7% 的多节律样本会“丢失部分信息”（例如同时有 SR 和 BIGU，被归到 OTHER 后 SR 信息不再体现在标签里）。


# 3 改进方向
## 3.1 SR再分类
虽然单次的 RR 间期分不清，但 RR 间期的波动（心率变异性, HRV） 在健康与带病之间是有显著差异的。
隐匿的信号：医学研究表明，患有心血管疾病（如冠心病、高血压肥厚）的患者，其心脏自主神经系统的调节能力会下降。

表现形式：

- 健康 SR：RR 间期会有细微的、富有弹性的变化（HRV 高）。
- 带病 SR：RR 间期往往表现得过于“呆板”或“僵硬”（HRV 降低）。

说服力所在：如果你做了二次标注，并发现模型能够识别出“带病 SR”，那么你就可以在论文中宣称：“本模型不仅能识别节律异常，还能通过 RR 间期的微小扰动（Micro-fluctuations）捕获心脏潜在的病理状态。” 这种论点在学术上非常高级。

## 3.2 具体的实现逻辑建议

建议不要把它们看作完全不同的类，而是看作 **“SR 的两个亚型”**。在 **PTB-XL** 中，可以利用 `scp_codes` 这样操作：

### 标签定义

- **SR_Healthy**  
  `scp_codes` **包含 `SR` 且包含 `NORM`**。

- **SR_Unhealthy**  
  `scp_codes` **包含 `SR` 但不包含 `NORM`**（说明存在 `MI`, `STTC`, `CD`, `HYP` 等问题）。

### 数据分布预估

在 **1.6 万个 SR 样本**中：

- 约 **9,000 个** 为 **SR_Healthy**
- 约 **7,000 个** 为 **SR_Unhealthy**


### 模型验证

如果 **RR-only 模型** 对这两类的区分度（**AUC**）能达到 **0.65 – 0.70** 以上，  
说明你的模型**确实捕捉到了 HRV 层面的病理特征**。