# SIGnature_OOD: OOD-Aware Representation and Trustworthy Attribution for Single-Cell RNA-seq

## 项目简介

SIGnature_OOD 是 SIGnature 的扩展架构，针对**分布外（Out-of-Distribution, OOD）样本**在单细胞 RNA 测序数据分析中的挑战，提供：

- **局部几何自适应**：在 Foundation Model 嵌入空间中进行 OOD 感知的几何微调
- **表征解耦**：将潜表征分解为通用生物学程序、上下文特异性程序和技术噪声三个子空间
- **结构化 OOD 检测**：多类别 OOD 概率输出（ID、通用生物学 OOD、上下文特异性 OOD、领域偏移 OOD、模型盲区）
- **转移 Token 分解**：方向-幅度解耦的程序级归因
- **可信归因验证**：四级归因体系与置信度评估
- **课程学习策略**：渐进式 OOD 引入，提升模型泛化能力

## 核心问题

单细胞 RNA-seq 数据分析中，OOD 样本（如罕见细胞类型、药物处理后的极端状态、技术批次差异）会导致：
- 模型预测置信度过高但不准确（Overconfidence）
- 归因解释不可靠（Unreliable Attribution）
- 生物学结论误导（Biological Misinterpretation）

## 架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SIGnature_OOD Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: scRNA-seq Expression Matrix                                         │
│         └───► [ Foundation Model Encoder ]                                  │
│                   └───► z_fm (原始嵌入)                                     │
│                         └───► [ Geometry Adapter ]                          │
│                               │  ├─ 对比损失 (Contrastive Loss)             │
│                               │  ├─ OOD 重建损失 (OOD Reconstruction)       │
│                               │  └─ 拓扑保持损失 (Topology Preservation)    │
│                               │  └─ 课程学习 (Curriculum Learning)          │
│                               └───► z_geo (几何自适应嵌入)                   │
│                                     └───► [ Representation Disentanglement ]│
│                                           │  ├─ z_shared_universal          │
│                                           │  ├─ z_shared_conditional        │
│                                           │  ├─ z_private                   │
│                                           │  └─ G×E 双线性交互              │
│                                           └───► [ Structured OOD Detector ] │
│                                                 │  ├─ ID                     │
│                                                 │  ├─ Generic Biology OOD    │
│                                                 │  ├─ Context-specific OOD   │
│                                                 │  ├─ Domain Shift OOD       │
│                                                 │  └─ Model Blind Region     │
│                                                 └───► [ Transition Token    │
│                                                       Decomposition ]        │
│                                                       │  ├─ Program Tokens     │
│                                                       │  ├─ Cross Attention    │
│                                                       │  ├─ Direction-Amplitude│
│                                                       │  └─ Scale Anchor Loss  │
│                                                       └───► [ Calibrated      │
│                                                             Classifier ]      │
│                                                             │  ├─ Source Clf      │
│                                                             │  ├─ Program Clf     │
│                                                             │  └─ Severity Clf    │
│                                                             └───► [ Extended     │
│                                                                   Integrated    │
│                                                                   Gradients ]   │
│                                                                   │  ├─ State    │
│                                                                   │  ├─ Transition│
│                                                                   │  ├─ Program  │
│                                                                   │  └─ Severity │
│                                                                   └───► [ Trust   │
│                                                                         System ]  │
│                                                                           │  ├─ L1 │
│                                                                           │  ├─ L2 │
│                                                                           │  ├─ L3 │
│                                                                           │  └─ L4 │
│                                                                           └───► Output│
│                                                                                 ├─ 基因归因分数│
│                                                                                 ├─ 四象限基因分类│
│                                                                                 ├─ OOD 概率分布│
│                                                                                 ├─ 转移程序分析│
│                                                                                 └─ 置信度报告│
└─────────────────────────────────────────────────────────────────────────────┘
```

## 训练流程（四阶段）

| 阶段 | 训练对象 | 冻结对象 | 核心损失 |
|------|---------|---------|---------|
| Stage 1 | Geometry Adapter | Foundation Model | 对比损失 + OOD重建损失 + 拓扑保持损失 |
| Stage 2 | Disentangler | Foundation Model + Geometry Adapter | 重建损失 + 域对抗损失 + 干预不变性损失 |
| Stage 3 | Transition Token + Attention + Scale | 前两阶段 | Scale Anchor Loss + 分类损失 |
| Stage 4 | Calibrated Classifier | 前三阶段 | 校准分类损失 |

## 核心功能

### 1. 局部几何自适应（Geometry Adapter）
- 仅微调 Foundation Model 最后几层
- 课程学习策略：渐进式引入 OOD 样本
- 支持多种调度策略：linear、exponential、cosine、staged
- 难度感知采样：优先训练简单 OOD，逐步引入困难 OOD

### 2. 表征解耦（Representation Disentanglement）
- 三空间分解：通用、条件、私有
- G×E 双线性交互层（基因-环境交互）
- 干预不变性损失：确保私有空间不含生物学信号

### 3. 结构化 OOD 检测
- 多类别 OOD 概率输出
- 支持自动 OOD 密度估计（kNN 方法）

### 4. 转移 Token 分解
- 程序级 Token 系统（毒性、衰老、技术噪声等）
- 方向-幅度解耦
- 用户自定义 Token 生物学初始化

### 5. 扩展归因体系
- 四级归因：状态归因、转移归因、程序归因、严重性归因
- 归因验证：多路径积分测试、注意力交叉验证、扰动梯度验证

### 6. 四象限基因分类
- 核心效应基因：状态+转移高分
- 纯驱动基因：转移高分、状态低分
- 状态标记基因：状态高分、转移低分
- 背景基因：双低分

## 快速开始

```python
import scanpy as sc
from SIGnature import SIGnature
from SIGnature.config import get_default_config

# 1. 加载数据
adata = sc.read_h5ad("your_data.h5ad")

# 2. 创建配置
config = get_default_config()

# 启用课程学习
config.model.representation.curriculum.enabled = True
config.model.representation.curriculum.schedule_type = "exponential"
config.model.representation.curriculum.max_ood_ratio = 0.5

# 3. 初始化模型
model = SIGnature(config=config)

# 4. 加载 Foundation Model
model.load_foundation_model(
    model_type="scsimilarity",
    model_path="./models/scsimilarity"
)

# 5. 训练（自动检测OOD）
model.fit(adata)

# 或手动指定OOD
# ood_mask = torch.tensor([False]*800 + [True]*200)
# model.fit(adata, ood_mask=ood_mask)

# 6. 推理
results = model.predict(adata)

# 7. 获取归因
attributions = model.attribute(adata)

# 8. 四象限基因分类
gene_classes = model.classify_genes(adata)

# 9. 获取置信度报告
trust_report = model.get_trust_report(adata)
```

## 配置说明

### 课程学习配置

```python
config.model.representation.curriculum.enabled = True      # 是否启用
config.model.representation.curriculum.schedule_type = "exponential"  # 调度类型
config.model.representation.curriculum.warmup_epochs = 5   # 预热轮数
config.model.representation.curriculum.ramp_up_epochs = 20 # 增长轮数
config.model.representation.curriculum.initial_ood_ratio = 0.05  # 初始比例
config.model.representation.curriculum.max_ood_ratio = 0.5 # 最大比例
config.model.representation.curriculum.difficulty_aware = True  # 难度感知
```

### 调度策略对比

| 策略 | 特点 | 适用场景 |
|------|------|---------|
| linear | 线性增长 | 数据分布均匀 |
| exponential | 前期快速增长 | OOD样本较多 |
| cosine | 平滑加速/减速 | 敏感数据 |
| staged | 离散阶段跳跃 | 多阶段训练 |

## 模型支持

| 模型 | 状态 | 权重路径 |
|------|------|---------|
| scSimilarity | ✅ 完整支持 | `models/scsimilarity/` |
| scFoundation | ⚠️ 需下载权重 | [HuggingFace](https://huggingface.co/genbio-ai/scFoundation) |
| scVI | ⚠️ 需下载权重 | [CZI Census](https://cellxgene.cziscience.com/census-models) |
| SSL | ⚠️ 需下载权重 | [HuggingFace](https://huggingface.co/TillR/sc_pretrained) |

## 测试

```bash
# 运行课程学习测试
python test_curriculum_learning.py

# 运行新功能测试
python test_new_features_independent.py

# 运行基础测试
python test_simple.py
```

## 与原版 SIGnature 的关系

| 特性 | 原版 SIGnature | SIGnature_OOD |
|------|---------------|---------------|
| 基因归因 | ✅ Integrated Gradients | ✅ 扩展四级归因 |
| Foundation Model | ✅ 支持四种模型 | ✅ 相同 |
| OOD 感知 | ❌ | ✅ 结构化 OOD 检测 |
| 表征解耦 | ❌ | ✅ 三空间分解 + G×E |
| 课程学习 | ❌ | ✅ 渐进式 OOD 引入 |
| 转移 Token | ❌ | ✅ 程序级分解 |
| 校准分类器 | ❌ | ✅ 三级分类 |
| 置信度系统 | ❌ | ✅ 四级信任体系 |
| 基因分类 | ❌ | ✅ 四象限分类 |
| 归因验证 | ❌ | ✅ 多方法验证 |
| 向后兼容 | - | ✅ 可独立运行原版流程 |

## 模块结构

```
src/SIGnature/
├── __init__.py
├── SIGnature.py              # Facade API
├── config.py                 # 配置系统
├── attribution/              # 归因模块
│   ├── extended_attribution.py      # 扩展归因
│   ├── attribution_validation.py    # 归因验证
│   └── gene_classification.py       # 四象限分类
├── core/                     # 核心抽象
│   └── base_wrapper.py       # 模型包装器基类
├── models/                   # Foundation Model 实现
│   ├── scsimilarity.py
│   ├── scfoundation.py
│   ├── scvi.py
│   └── ssl.py
├── pipeline/                 # 训练/推理管道
│   ├── train.py              # 四阶段训练
│   ├── inference.py          # 推理
│   └── evaluation.py         # 评估
├── representation/           # 表征模块
│   ├── geometry_adaptation.py       # 几何自适应
│   ├── curriculum_scheduler.py      # 课程学习调度器
│   ├── disentanglement.py           # 表征解耦
│   └── ood_detector.py              # OOD检测
├── transition/               # 转移模块
│   ├── program_token.py             # 程序Token
│   ├── token_decomposition.py       # Token分解
│   ├── attention.py                 # 交叉注意力
│   └── scale_predictor.py           # 尺度预测
├── trust/                    # 信任模块
│   ├── calibrated_classifier.py     # 校准分类器
│   └── confidence.py                # 置信度系统
└── results/                  # 结果输出
    ├── interpretation_result.py
    ├── prediction_result.py
    └── trust_report.py
```

## 许可证

MIT License

## 引用

如需引用 SIGnature，请使用：

**Scoring gene importance by interpreting single-cell foundation models.**
*Maxwell P. Gold et al.*, Nature Biotechnology (2026).

---

> **注意**：SIGnature_OOD 是 SIGnature 的扩展研究项目，用于探索 OOD 感知的单细胞数据分析方法。详细实现状态和下一步计划请参考 [HANDOFF.md](HANDOFF.md)。
