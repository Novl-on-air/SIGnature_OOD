<div align="center">

## 🧬 SIGnature_OOD: OOD-Aware Representation and Trustworthy Attribution for Single-Cell RNA-seq

**🔍 你的单细胞数据分析助手——面向分布外挑战的可信归因系统。**

[概述](#-概述) • [核心功能](#-核心功能) • [快速开始](#-快速开始) • [训练流程](#-训练流程) • [模型支持](#-模型支持) • [与原版对比](#-与原版对比) • [测试](#-测试) • [引用](#-引用)

</div>

---

## 💡 概述

SIGnature_OOD 是 SIGnature 的扩展架构，针对**分布外（Out-of-Distribution, OOD）样本**在单细胞 RNA 测序数据分析中的挑战提供完整解决方案。

它首先通过几何自适应微调 Foundation Model 嵌入空间，再将表征解耦为通用生物学程序、上下文特异性程序和技术噪声三个子空间，最后通过结构化 OOD 检测、转移 Token 分解和四级归因验证，输出可信的生物学解释。

---

## ✨ 核心功能

<table align="center" width="100%" style="border: none; table-layout: fixed;">
<tr>
<td width="25%" align="center" style="vertical-align: top; padding: 16px;">

### 📐 **几何自适应**

在 Foundation Model 嵌入空间中进行局部几何微调，仅调整 OOD 附近的表征几何，不改变整体流形结构。支持课程学习策略，渐进式引入 OOD 样本。

</td>
<td width="25%" align="center" style="vertical-align: top; padding: 16px;">

### 🧩 **表征解耦**

将潜表征分解为通用、条件、私有三个子空间，实现 G×E 交互建模和不对称解耦，确保私有空间不含生物学信号。

</td>
<td width="25%" align="center" style="vertical-align: top; padding: 16px;">

### 🌐 **结构化 OOD 检测**

多类别 OOD 概率输出：ID、通用生物学 OOD、上下文特异性 OOD、领域偏移 OOD、模型盲区。支持自动 OOD 密度估计。

</td>
<td width="25%" align="center" style="vertical-align: top; padding: 16px;">

### 🔄 **转移 Token 分解**

方向-幅度解耦的程序级归因，支持用户自定义 Token 生物学初始化和验证，输出 Δ_direction、Scale、Δ_pred 和 Attention Weight。

</td>
</tr>
</table>

<table align="center" width="100%" style="border: none; table-layout: fixed;">
<tr>
<td width="33%" align="center" style="vertical-align: top; padding: 16px;">

### 📊 **扩展归因体系**

四级归因：状态归因、转移归因、程序归因、严重性归因。保留原始 SIGnature 归因结果用于对比。

</td>
<td width="33%" align="center" style="vertical-align: top; padding: 16px;">

### ✅ **归因验证**

多路径积分测试、注意力交叉验证、扰动梯度验证，建立四级独立信任体系（L1-L4）。

</td>
<td width="33%" align="center" style="vertical-align: top; padding: 16px;">

### 🎯 **四象限基因分类**

基于状态和转移分数的基因分类：核心效应基因、纯驱动基因、状态标记基因、背景基因。

</td>
</tr>
</table>

---

## 🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/genentech/signature.git
cd signature
pip install -e .
```

### 2. 运行

```python
import scanpy as sc
from SIGnature import SIGnature
from SIGnature.config import get_default_config

# 加载数据
adata = sc.read_h5ad("your_data.h5ad")

# 创建配置（启用课程学习）
config = get_default_config()
config.model.representation.curriculum.enabled = True
config.model.representation.curriculum.schedule_type = "exponential"

# 初始化模型
model = SIGnature(config=config)
model.load_foundation_model(model_type="scsimilarity", model_path="./models/scsimilarity")

# 训练
model.fit(adata)

# 推理与分析
results = model.predict(adata)
attributions = model.attribute(adata)
gene_classes = model.classify_genes(adata)
trust_report = model.get_trust_report(adata)
```

---

## 📋 训练流程（四阶段）

<table align="center" width="100%" style="border: 1px solid #ddd; border-collapse: collapse;">
<tr style="background-color: #f8f9fa;">
<th style="border: 1px solid #ddd; padding: 12px; text-align: center;">阶段</th>
<th style="border: 1px solid #ddd; padding: 12px; text-align: center;">训练对象</th>
<th style="border: 1px solid #ddd; padding: 12px; text-align: center;">冻结对象</th>
<th style="border: 1px solid #ddd; padding: 12px; text-align: center;">核心损失</th>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 12px; text-align: center;">Stage 1</td>
<td style="border: 1px solid #ddd; padding: 12px; text-align: center;">Geometry Adapter</td>
<td style="border: 1px solid #ddd; padding: 12px; text-align: center;">Foundation Model</td>
<td style="border: 1px solid #ddd; padding: 12px; text-align: center;">对比损失 + OOD重建损失 + 拓扑保持损失</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 12px; text-align: center;">Stage 2</td>
<td style="border: 1px solid #ddd; padding: 12px; text-align: center;">Disentangler</td>
<td style="border: 1px solid #ddd; padding: 12px; text-align: center;">FM + Geometry Adapter</td>
<td style="border: 1px solid #ddd; padding: 12px; text-align: center;">重建损失 + 域对抗损失 + 干预不变性损失</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 12px; text-align: center;">Stage 3</td>
<td style="border: 1px solid #ddd; padding: 12px; text-align: center;">Token + Attention + Scale</td>
<td style="border: 1px solid #ddd; padding: 12px; text-align: center;">前两阶段</td>
<td style="border: 1px solid #ddd; padding: 12px; text-align: center;">Scale Anchor Loss + 分类损失</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 12px; text-align: center;">Stage 4</td>
<td style="border: 1px solid #ddd; padding: 12px; text-align: center;">Calibrated Classifier</td>
<td style="border: 1px solid #ddd; padding: 12px; text-align: center;">前三阶段</td>
<td style="border: 1px solid #ddd; padding: 12px; text-align: center;">校准分类损失</td>
</tr>
</table>

---

## 🧪 模型支持

- **scSimilarity**: 预训练权重包含在 `models/scsimilarity/`
- **scFoundation**: 权重可从 [HuggingFace](https://huggingface.co/genbio-ai/scFoundation/tree/main) 下载
- **scVI**: 权重可从 [CZI Census](https://cellxgene.cziscience.com/census-models) 下载
- **SSL-scTab**: 权重可从 [HuggingFace](https://huggingface.co/TillR/sc_pretrained/tree/main) 下载

---

## 🔄 与原版 SIGnature 的对比

<table align="center" width="100%" style="border: 1px solid #ddd; border-collapse: collapse;">
<tr style="background-color: #f8f9fa;">
<th style="border: 1px solid #ddd; padding: 8px;">特性</th>
<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">原版 SIGnature</th>
<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">SIGnature_OOD</th>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">基因归因</td>
<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Integrated Gradients</td>
<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">扩展四级归因</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">Foundation Model</td>
<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">4 种模型</td>
<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">相同</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">OOD 感知</td>
<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">❌</td>
<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">✅ 结构化 OOD 检测</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">表征解耦</td>
<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">❌</td>
<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">✅ 三空间分解 + G×E</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">课程学习</td>
<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">❌</td>
<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">✅ 渐进式 OOD 引入</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">转移 Token</td>
<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">❌</td>
<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">✅ 程序级分解</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">校准分类器</td>
<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">❌</td>
<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">✅ 三级分类</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">置信度系统</td>
<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">❌</td>
<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">✅ 四级信任体系</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">基因分类</td>
<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">❌</td>
<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">✅ 四象限分类</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">归因验证</td>
<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">❌</td>
<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">✅ 多方法验证</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">向后兼容</td>
<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">-</td>
<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">✅ 可独立运行原版流程</td>
</tr>
</table>

---

## 🧪 测试

```bash
# 课程学习测试
python test_curriculum_learning.py

# 新功能测试
python test_new_features_independent.py

# 基础测试
python test_simple.py
```

---

## 📝 引用

**Scoring gene importance by interpreting single-cell foundation models.**
*Maxwell P. Gold et al.*, Nature Biotechnology (2026).
https://www.nature.com/articles/s41587-026-03112-5

---

## 📜 许可证

MIT License
