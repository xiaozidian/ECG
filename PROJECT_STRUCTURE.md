# CycleVLA / OpenVLA 代码库结构说明

本文档详细解析了 `cyclevla_code` 文件夹下的文件结构和功能，帮助开发者快速上手。

## 1. 根目录文件

| 文件名 | 作用 |
| :--- | :--- |
| `quickstart.py` | **演示脚本**。加载预训练模型，对单样本数据进行推理，演示如何生成机器人动作。 |
| `pyproject.toml` | Python 项目配置文件，定义了项目依赖、构建系统等信息。 |
| `README.md` | 项目总览说明文档。 |
| `ALOHA.md`, `LIBERO.md` | 针对特定机器人环境（ALOHA, LIBERO）的详细文档。 |

## 2. 核心库: `prismatic/`

这是本项目的核心包，包含了 OpenVLA 模型的定义、训练逻辑和数据处理代码。

### 2.1 模型定义 (`prismatic/models/`)

| 目录/文件 | 作用 |
| :--- | :--- |
| `vlas/openvla.py` | **OpenVLA 模型核心类**。继承自 `PrismaticVLM`，增加了动作预测 (`predict_action`) 和 Tokenizer 逻辑。 |
| `vlms/prismatic.py` | 通用 VLM (Vision-Language Model) 基类。定义了视觉编码器和 LLM 如何结合。 |
| `backbones/` | 包含视觉（Vision）和语言（LLM）的主干网络定义。 |
| &nbsp;&nbsp;`vision/siglip_vit.py` | SigLIP 视觉编码器实现（OpenVLA 默认使用）。 |
| &nbsp;&nbsp;`llm/llama2.py` | Llama-2 LLM 实现（OpenVLA 基于 Llama-2 7B）。 |
| `action_heads.py` | 定义不同的动作预测头（如 L1 回归头、Diffusion 头）。 |
| `projectors.py` | 定义投影层（Projectors），用于将视觉/本体特征映射到 LLM 嵌入空间。 |
| `materialize.py` | 工厂模式文件，负责根据配置字符串实例化具体的 Backbone 和 VLM 模型。 |
| `load.py` | 提供从 Hugging Face 或本地加载预训练模型的接口。 |

### 2.2 VLA 特定逻辑 (`prismatic/vla/`)

| 文件名 | 作用 |
| :--- | :--- |
| `action_tokenizer.py` | **动作离散化工具**。负责将连续的机器人动作 (Continuous Actions) 转换为离散的 Token ID，反之亦然。 |
| `constants.py` | 定义关键常量，如动作维度 (`ACTION_DIM=7`)、归一化方式、机器人平台参数 (ALOHA/LIBERO)。 |
| `datasets/rlds/` | **RLDS 数据集处理**。包含用于加载 Open X-Embodiment 数据集的逻辑。 |
| &nbsp;&nbsp;`dataset.py` | 构建 TFDS/RLDS 数据集管道的主入口。 |
| &nbsp;&nbsp;`oxe/mixtures.py` | 定义不同数据集的混合比例（如 Bridge, RT-1 等）。 |
| &nbsp;&nbsp;`oxe/transforms.py` | 定义针对不同数据集的标准化变换逻辑。 |

### 2.3 训练与配置 (`prismatic/training/`, `prismatic/conf/`)

| 文件名 | 作用 |
| :--- | :--- |
| `training/strategies/fsdp.py` | FSDP (Fully Sharded Data Parallel) 训练策略实现，用于多卡高效训练。 |
| `training/train_utils.py` | 训练辅助函数，如计算 L1 Loss、动作掩码 (`action_mask`) 生成等。 |
| `extern/hf/` | **Hugging Face 集成**。包含自定义的 Config (`configuration_prismatic.py`) 和 Model (`modeling_prismatic.py`) 类，使得模型能通过 `AutoModel.from_pretrained` 加载。 |

## 3. 实验与评估: `experiments/`

包含针对不同机器人平台的评估脚本。

| 目录/文件 | 作用 |
| :--- | :--- |
| `robot/openvla_utils.py` | **核心评估工具库**。封装了模型加载 (`get_vla`)、图像预处理、动作推理 (`get_vla_action`) 等高频函数。 |
| `robot/robot_utils.py` | 通用机器人工具函数（如坐标转换、随机种子设置）。 |
| `robot/libero/` | **LIBERO 仿真环境**相关代码。 |
| &nbsp;&nbsp;`run_libero_eval.py` | LIBERO 评估主脚本。运行此脚本可在仿真环境中测试模型性能。 |
| &nbsp;&nbsp;`libero_utils.py` | LIBERO 环境初始化、图像提取等辅助函数。 |
| `robot/aloha/` | **ALOHA 真实机器人**相关代码。 |
| &nbsp;&nbsp;`run_aloha_eval.py` | ALOHA 评估脚本。 |

## 4. 脚本工具: `vla-scripts/`

提供模型微调和部署的顶层工作流脚本。

| 文件名 | 作用 |
| :--- | :--- |
| `finetune.py` | **微调主脚本**。使用 LoRA (Low-Rank Adaptation) 对 OpenVLA 进行微调。支持多卡训练。 |
| `deploy.py` | **部署服务脚本**。启动一个 FastAPI 服务器，提供 HTTP 接口 (`/act`) 供客户端调用模型进行推理。 |
| `extern/convert_openvla_weights_to_hf.py` | 权重转换脚本。将训练时的 Checkpoint 转换为标准的 Hugging Face 格式以便发布。 |

## 总结：阅读建议

1.  **想跑通 Demo**: 从 `quickstart.py` 开始。
2.  **想理解模型架构**: 阅读 `prismatic/models/vlas/openvla.py` 和 `prismatic/vla/action_tokenizer.py`。
3.  **想进行微调**: 重点看 `vla-scripts/finetune.py` 和 `prismatic/vla/datasets/rlds/dataset.py`（数据加载）。
4.  **想做评估**: 参考 `experiments/robot/libero/run_libero_eval.py` 了解模型是如何与环境交互的。
