# SIGnature 项目交接文档

## 一、任务背景与目标

### 1.1 原始问题
SIGnature是一个基于积分梯度(IG)的单细胞归因框架，将基因表达向量输入预训练Foundation Model，计算隐空间表征的归因分数，识别生物学状态下的关键调控基因。

**核心问题**：SIGnature在分布外(OOD)样本上存在系统性失效：
- Foundation Model学到的细胞状态流形有限，仅覆盖常见细胞状态
- 药物处理或扰动导致的异常细胞落在流形外部或稀疏区域
- 积分梯度依赖表征质量有问题的隐空间，输出生物学无意义的数学噪声
- 缺乏适用域边界判断能力，无法识别模型盲区

### 1.2 解决方案：OOD-aware表示几何自适应 + Token-based转移分解

**核心命题**：解决SIGnature在OOD样本上的失效，不能仅改进归因算法，而必须：
1. 通过局部表征几何自适应调整FM的表示空间
2. 在此基础上实现结构化OOD检测
3. 实现可归因的状态转移建模
4. 实现归因驱动的细胞分类

**最终目标**：不是拒识OOD样本，而是通过归因将OOD翻译为已知生物学程序的组合，保留发现能力的同时控制可解释性边界。

### 1.3 四级信任层级体系
| 层级 | 评估模块 | 评估内容 |
|------|----------|----------|
| L1 | FM编码层 | 细胞密度、近邻距离、重构误差 |
| L2 | 解耦模块 | 五种OOD类型概率分布及不确定性 |
| L3 | 转移分解模块 | Δ_pred解释方差比例、缩放因子不确定性 |
| L4 | 归因模块 | 多路径一致性、注意力交叉验证、扰动梯度验证 |

---

## 二、已完成的工作

### 2.1 架构框架（100%）

#### 核心模块
| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| Facade API | `SIGnature.py` | ✅ | 统一用户接口 |
| 配置系统 | `config.py` | ✅ | YAML配置管理 |
| BaseModelWrapper | `core/base_wrapper.py` | ✅ | Foundation Model统一接口 |
| 模型工厂 | `models/model_factory.py` | ✅ | 模型动态加载 |

#### Foundation Model Wrappers
| 模型 | 文件 | 状态 |
|------|------|------|
| scSimilarity | `models/scsimilarity.py` | ✅ 已测试 |
| scFoundation | `models/scfoundation.py` | ✅ 代码就绪 |
| scVI | `models/scvi.py` | ✅ 代码就绪 |
| SSL | `models/ssl.py` | ✅ 代码就绪 |

#### 核心功能模块
| 模块 | 文件 | 状态 |
|------|------|------|
| 局部几何自适应 | `representation/geometry_adaptation.py` | ✅ |
| 课程学习调度器 | `representation/curriculum_scheduler.py` | ✅ |
| 表征解耦（含双线性层） | `representation/disentanglement.py` | ✅ |
| 结构化OOD检测 | `representation/ood_detector.py` | ✅ |
| 转移Token分解 | `transition/token_decomposition.py` | ✅ |
| Token生物学初始化 | `transition/program_token.py` | ✅ 用户可自定义 |
| 校准分类器 | `trust/calibrated_classifier.py` | ✅ |
| 置信度系统 | `trust/confidence.py` | ✅ |
| 扩展归因 | `attribution/extended_attribution.py` | ✅ |
| 归因验证体系 | `attribution/attribution_validation.py` | ✅ |
| 四象限基因分类 | `attribution/gene_classification.py` | ✅ |

#### 管道流程
| 管道 | 文件 | 状态 |
|------|------|------|
| 训练管道（四阶段） | `pipeline/train.py` | ✅ |
| 推理管道 | `pipeline/inference.py` | ✅ |
| 评估管道 | `pipeline/evaluation.py` | ✅ |

### 2.2 已实现的关键功能

#### 1. 双线性层（G×E交互）
- 位置：`representation/disentanglement.py`
- 实现：`z_shared_conditional = torch.einsum('nj,jlk,nl->nk', h_cell, W, h_domain)`
- 作用：显式建模细胞状态与域背景的乘法交互

#### 2. 干预不变性损失
- 位置：`representation/disentanglement.py`
- 实现：`L_private_invariance = ||z_private(control) - z_private(drug)||`
- 作用：强制私有空间不包含生物学响应

#### 3. Token生物学初始化（用户可自定义）
- 位置：`transition/program_token.py`
- 新增API：
  - `add_token_category()` - 添加新类别
  - `add_token_to_category()` - 向类别添加Token
  - `set_reference_genes()` - 设置参考基因
  - `set_reference_genes_for_category()` - 为类别批量设置参考基因
  - `initialize_with_biology_reference(custom_init_fn=...)` - 自定义初始化
  - `initialize_from_gene_expression()` - 从基因表达初始化
  - `validate_tokens()` - Token验证
  - `get_token_quality_report()` - Token质量报告

#### 4. 归因验证体系
- 位置：`attribution/attribution_validation.py`
- 实现：
  - 多路径积分检验（线性/曲线/Z字形路径）
  - 注意力权重交叉验证
  - 扰动梯度验证
  - 综合置信度评估

#### 5. 四象限基因分类
- 位置：`attribution/gene_classification.py`
- 分类逻辑：
  - 核心效应基因：状态高+转移高
  - 纯驱动基因：状态低+转移高
  - 状态标记基因：状态高+转移低
  - 背景基因：状态低+转移低

### 2.3 训练流程
```
Stage 1 → Stage 2 → Stage 3 → Stage 4
  ↓         ↓         ↓         ↓
Geometry → Disentangle → Transition → Classification
Adapter   ment        Learning    & Attribution
```

### 2.4 测试验证（全部通过 ✅）
- ✅ 双线性层计算正确
- ✅ 干预不变性损失计算正确
- ✅ Token生物学初始化（默认/自定义）
- ✅ Token验证与质量报告
- ✅ 归因验证体系（多路径一致性1.0000，扰动一致性0.4200）
- ✅ 四象限基因分类（核心效应24-27%，纯驱动24-27%，状态标记24-27%，背景22-25%）
- ✅ 用户自定义Token类别和基因

---

## 三、当前卡住的地方

### 3.1 Foundation Model加载问题（Windows环境）
- **问题**：在Windows环境下，`load_foundation_model()` 调用失败
- **错误信息**：`ValueError: Unknown model type: scsimilarity`
- **原因**：Python模块导入路径问题，Windows路径分隔符与模块导入机制冲突
- **影响**：无法使用真实Foundation Model进行端到端测试
- **临时方案**：使用独立测试脚本验证各模块功能，绕过Foundation Model加载

### 3.2 模型文件完整性
| 模型 | 状态 | 缺失文件 |
|------|------|----------|
| scSimilarity | ✅ 完整 | - |
| scFoundation | ❌ 不完整 | `models.ckpt` |
| scVI | ❌ 不完整 | 完整的scVI模型目录 |
| SSL | ❌ 不完整 | 完整的SSL模型权重 |

### 3.3 待实现的功能（方案要求但未完成）
| 功能 | 优先级 | 说明 |
|------|--------|------|
| 课程学习策略 | ✅ 已完成 | Geometry Adapter的渐进式OOD引入（支持linear/exponential/cosine/staged调度，难度感知采样） |
| 原始FM副本保留 | 中 | 微调前后归因对比 |
| Token留出验证 | 中 | 使用独立数据验证Token质量 |
| 自适应基线策略 | 中 | 三级基线策略（对照>参考图谱>拒绝） |
| 单细胞GSEA融合 | 低 | 通路富集分析 |
| 七项诚实报告 | 低 | Token质量、转移解释充分度等报告 |

---

## 四、下一步计划

### 4.1 短期目标（1-2周）
1. **解决Foundation Model加载问题**
   - 修复Windows环境下的模块导入问题
   - 确保scSimilarity模型能正常加载

2. **添加课程学习策略**
   - 在Geometry Adapter中实现渐进式OOD引入
   - 先在ID数据上微调，再逐步引入OOD数据

3. **保留原始FM副本**
   - 在微调前保存原始FM权重
   - 推理时同时输出原始和微调后的归因结果

### 4.2 中期目标（2-4周）
1. **实现Token留出验证**
   - 使用独立扰动数据验证Token区分能力
   - 验证不通过的Token降低权重或标记

2. **实现自适应基线策略**
   - 三级基线策略：对照数据>参考图谱>拒绝
   - 跨平台偏移自动补偿

3. **完善四阶段训练的Loss配置**
   - 确保每个阶段的Loss权重配置正确
   - 添加Early Stopping策略（已有基础实现）

### 4.3 长期目标（4周+）
1. **实现单细胞GSEA融合**
   - 归因驱动的通路富集分析
   - 与已知毒性程序库比对

2. **实现七项诚实报告**
   - Token质量报告
   - 转移解释充分度报告
   - 跨平台偏移检测报告
   - 局部几何自适应效果报告
   - 微调前后归因对比报告

---

## 五、踩过的坑与注意事项

### 5.1 代码实现坑

#### 1. 双线性层维度计算
- **问题**：`torch.matmul`维度匹配困难，无法直接处理batch
- **解决方案**：使用`torch.einsum`实现批处理的双线性交互
- **关键代码**：`z_shared_conditional = torch.einsum('nj,jlk,nl->nk', h_cell, W, h_domain)`

#### 2. 积分梯度backward错误
- **问题**：`pred.backward()`要求标量输出，直接调用报错
- **解决方案**：使用`pred.mean().backward()`确保标量输出
- **位置**：`attribution_validation.py`

#### 3. torch.sin输入类型
- **问题**：`torch.sin()`要求Tensor输入，不能是float
- **解决方案**：`torch.sin(torch.tensor(t * np.pi))`
- **位置**：`attribution_validation.py`的路径函数

#### 4. Windows路径问题
- **问题**：Windows路径分隔符`\`与Python导入机制冲突
- **解决方案**：使用`os.path.join()`构建路径，避免硬编码路径

#### 5. 属性名大小写问题
- **问题**：测试脚本中使用`BIOLOGY_REFERENCE_GENES`（大写），实际属性是`biology_reference_genes`（小写）
- **教训**：属性命名要保持一致性，测试前检查属性名

### 5.2 架构设计坑

#### 1. Token初始化的生物学意义
- **问题**：随机初始化的Token缺乏生物学意义
- **解决方案**：提供用户自定义参考基因和初始化函数的接口

#### 2. 模块间维度匹配
- **问题**：不同Foundation Model输出维度不同（scSimilarity: 128, scFoundation: 3072）
- **解决方案**：动态检测嵌入维度，在`load_foundation_model`后重新初始化模块

#### 3. 双线性层参数规模
- **问题**：`W: [shared_dim, shared_dim, shared_dim]`参数规模较大
- **解决方案**：使用低秩分解或限制shared_dim大小

### 5.3 训练策略坑

#### 1. 学习率设置
- Geometry Adapter的学习率应远小于预训练学习率（1/100或更低）
- 使用早停策略防止过拟合

#### 2. Loss权重平衡
- 多Loss组合时需要调整权重
- 建议先单独训练每个Loss，再逐步组合

#### 3. 数据准备
- 需要配对的control-drug数据计算干预不变性损失
- 多样性约束：覆盖多种药物、剂量、时间点、细胞类型

---

## 六、项目结构

```
SIGnature-main/
├── src/SIGnature/
│   ├── SIGnature.py                    # Facade API
│   ├── config.py                       # 配置系统
│   ├── core/
│   │   └── base_wrapper.py             # Foundation Model基类
│   ├── models/
│   │   ├── model_factory.py            # 模型工厂
│   │   ├── scsimilarity.py             # scSimilarity包装器
│   │   ├── scfoundation.py             # scFoundation包装器
│   │   ├── scvi.py                     # scVI包装器
│   │   └── ssl.py                      # SSL包装器
│   ├── representation/
│   │   ├── geometry_adaptation.py      # 几何适配器
│   │   ├── disentanglement.py          # 表示解耦（含双线性层）
│   │   └── ood_detector.py             # OOD检测
│   ├── transition/
│   │   ├── program_token.py            # Token库（用户可自定义）
│   │   ├── token_decomposition.py      # 转换学习
│   │   ├── attention.py                # 注意力机制
│   │   ├── entmax.py                   # Entmax稀疏注意力
│   │   ├── scale_predictor.py          # 缩放因子预测
│   │   └── transition_pipeline.py      # 转移管道
│   ├── trust/
│   │   ├── confidence.py               # 信任系统
│   │   └── calibrated_classifier.py    # 校准分类器
│   ├── attribution/
│   │   ├── extended_attribution.py     # 扩展归因
│   │   ├── attribution_validation.py   # 归因验证
│   │   ├── gene_classification.py      # 四象限基因分类
│   │   ├── engine.py                   # 归因引擎
│   │   └── normalization.py            # 归因归一化
│   ├── pipeline/
│   │   ├── base.py                     # 管道基类
│   │   ├── train.py                    # 训练流程（四阶段）
│   │   ├── inference.py                # 推理流程
│   │   └── evaluation.py               # 评估流程
│   ├── results/
│   │   ├── prediction_result.py        # 预测结果对象
│   │   ├── interpretation_result.py    # 解释结果对象
│   │   └── trust_report.py             # 信任报告对象
│   ├── io/
│   │   ├── tiledb.py                   # TileDB IO
│   │   └── dataframe_utils.py          # 数据帧工具
│   ├── data/
│   │   └── preprocessing.py            # 数据预处理
│   ├── plotting/
│   │   └── meta_plots.py               # 可视化工具
│   ├── meta.py                         # Meta类（DataFrame包装）
│   └── utils.py                        # 工具函数
├── models/                             # Foundation Model权重
│   ├── scsimilarity/                   # ✅ 完整（encoder.ckpt, gene_order.tsv等）
│   ├── scfoundation/                   # ❌ 缺models.ckpt
│   ├── scvi/                           # ❌ 不完整（仅有ensembl_gene_symbols.txt）
│   └── ssl/                            # ❌ 不完整（仅有var.parquet）
├── test_new_features_independent.py    # 新功能独立测试（全部通过）
├── test_custom_tokens.py               # Token自定义测试
└── HANDOFF.md                          # 本交接文档
```

---

## 七、快速开始

```python
from SIGnature import SIGnature
from SIGnature.config import get_default_config

# 创建模型
config = get_default_config()
model = SIGnature(config=config)

# 加载Foundation Model（需先解决Windows加载问题）
model.load_foundation_model(model_type="scsimilarity", model_path="./models/scsimilarity")

# 训练（四阶段）
model.fit(adata)

# 推理
predictions = model.predict(adata)

# 归因分析
interpretations = model.interpret(adata)

# 自定义Token
from SIGnature.transition.program_token import ProgramTokenBank, DEFAULT_TOKEN_CATEGORIES, DEFAULT_BIOLOGY_REFERENCE_GENES

token_bank = ProgramTokenBank(
    token_dim=64,
    token_categories={"my_category": ["my_token1", "my_token2"]},
    biology_reference_genes={"my_token1": ["GENE1", "GENE2"]}
)
token_bank.add_token_category("new_category", ["token_a", "token_b"])
token_bank.set_reference_genes("token_a", ["GENEA", "GENEB"])
token_bank.initialize_with_biology_reference(gene_order=gene_list)
```

---

## 八、关键文件说明

| 文件 | 核心作用 | 关键类/函数 |
|------|----------|-------------|
| `SIGnature.py` | 用户入口 | `load_foundation_model()`, `fit()`, `predict()`, `interpret()` |
| `config.py` | 配置管理 | `SIGnatureConfig`, `get_default_config()` |
| `base_wrapper.py` | FM统一接口 | `BaseModelWrapper.encode()` |
| `geometry_adaptation.py` | 几何微调 | `GeometryAdapter`, 对比学习/拓扑保持 |
| `disentanglement.py` | 表示解耦 | `Disentangler`(双线性层), `intervention_invariance_loss` |
| `program_token.py` | Token管理 | `ProgramTokenBank`, 用户自定义API, `DEFAULT_TOKEN_CATEGORIES`, `DEFAULT_BIOLOGY_REFERENCE_GENES` |
| `token_decomposition.py` | 转移分解 | `TokenDecomposer`, Entmax注意力 |
| `attribution_validation.py` | 归因验证 | `AttributionValidator`, 三重验证 |
| `gene_classification.py` | 基因分类 | `GeneQuadrantClassifier`, 四象限分类 |
| `pipeline/train.py` | 四阶段训练 | `TrainingPipeline`, `EarlyStopping` |
| `pipeline/inference.py` | 推理流程 | `InferencePipeline`, `predict()`, `interpret()` |

---

## 九、运行测试

```bash
# 独立功能测试（全部通过 ✅）
python test_new_features_independent.py

# Token自定义测试
python test_custom_tokens.py
```

---

## 十、联系方式与资源

- **方案文档**：`方案内容.txt`（由用户提供的方案.docx转换）
- **模型文件位置**：`./models/`目录下
- **测试脚本**：`test_custom_tokens.py`, `test_new_features_independent.py`
- **核心依赖**：PyTorch, NumPy, Pandas, scanpy, torch-scatter

---

*文档生成日期：2026-07-13*
*项目状态：架构框架完整，功能模块基本实现，待解决Foundation Model加载问题和部分方案要求功能*
*测试状态：全部独立功能测试通过 ✅*