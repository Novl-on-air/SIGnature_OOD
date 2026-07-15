# SIGnature_OOD: OOD-Aware Representation and Trustworthy Attribution for Single-Cell RNA-seq

================================================================================

**SIGnature_OOD** 是 SIGnature 的扩展架构，针对**分布外（Out-of-Distribution, OOD）样本**在单细胞 RNA 测序数据分析中的挑战，提供局部几何自适应、表征解耦、结构化 OOD 检测、转移 Token 分解、可信归因验证和课程学习策略。

Documentation
--------------------------------------------------------------------------------

详细文档和教程请参考：
https://genentech.github.io/SIGnature/index.html


Download & Install
--------------------------------------------------------------------------------

项目代码可以通过以下方式下载和安装：

```bash
git clone https://github.com/genentech/signature.git
cd signature
pip install -e .
```


Core Architecture
--------------------------------------------------------------------------------

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SIGnature_OOD Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: scRNA-seq Expression Matrix                                         │
│         └───► [ Foundation Model Encoder ]                                  │
│                   └───► z_fm (原始嵌入)                                     │
│                         └───► [ Geometry Adapter ]                          │
│                               │  ├─ Contrastive Loss                        │
│                               │  ├─ OOD Reconstruction Loss                 │
│                               │  └─ Topology Preservation Loss              │
│                               │  └─ Curriculum Learning                     │
│                               └───► z_geo (几何自适应嵌入)                   │
│                                     └───► [ Representation Disentanglement ]│
│                                           │  ├─ z_shared_universal          │
│                                           │  ├─ z_shared_conditional        │
│                                           │  ├─ z_private                   │
│                                           │  └─ G×E Bilinear Interaction    │
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
│                                                                                 ├─ Attribution│
│                                                                                 ├─ Gene Classification│
│                                                                                 ├─ OOD Probability│
│                                                                                 ├─ Transition Program│
│                                                                                 └─ Trust Report│
└─────────────────────────────────────────────────────────────────────────────┘
```


Four-Stage Training Process
--------------------------------------------------------------------------------

| Stage | Train | Freeze | Core Loss |
|-------|-------|--------|-----------|
| Stage 1 | Geometry Adapter | Foundation Model | Contrastive + OOD Reconstruction + Topology |
| Stage 2 | Disentangler | FM + Geometry Adapter | Reconstruction + Domain Adversarial + Intervention Invariance |
| Stage 3 | Token + Attention + Scale | First two stages | Scale Anchor Loss + Classification |
| Stage 4 | Calibrated Classifier | First three stages | Calibration Loss |


Key Features
--------------------------------------------------------------------------------

1. **Geometry Adapter**: Local geometry adaptive fine-tuning with curriculum learning

2. **Representation Disentanglement**: Decompose latent space into universal, conditional, and private subspaces with G×E interaction

3. **Structured OOD Detection**: Multi-category OOD probability output (ID, Generic Biology OOD, Context-specific OOD, Domain Shift OOD, Model Blind Region)

4. **Transition Token Decomposition**: Direction-amplitude decoupled program-level attribution

5. **Extended Attribution**: Four-level attribution (State, Transition, Program, Severity)

6. **Gene Quadrant Classification**: Core effect, pure driver, state marker, and background genes


Quick Start
--------------------------------------------------------------------------------

```python
import scanpy as sc
from SIGnature import SIGnature
from SIGnature.config import get_default_config

# Load data
adata = sc.read_h5ad("your_data.h5ad")

# Create config
config = get_default_config()
config.model.representation.curriculum.enabled = True
config.model.representation.curriculum.schedule_type = "exponential"

# Initialize model
model = SIGnature(config=config)

# Load foundation model
model.load_foundation_model(model_type="scsimilarity", model_path="./models/scsimilarity")

# Train
model.fit(adata)

# Inference
results = model.predict(adata)

# Attribution
attributions = model.attribute(adata)

# Gene classification
gene_classes = model.classify_genes(adata)

# Trust report
trust_report = model.get_trust_report(adata)
```


Model Support
--------------------------------------------------------------------------------

1. `scSimilarity`: pretrained weights included in `models/scsimilarity/`
2. `scFoundation`: weights can be downloaded from https://huggingface.co/genbio-ai/scFoundation/tree/main
3. `scVI`: weights can be downloaded from https://cellxgene.cziscience.com/census-models
4. `SSL-scTab`: weights can be downloaded from https://huggingface.co/TillR/sc_pretrained/tree/main


Testing
--------------------------------------------------------------------------------

```bash
python test_curriculum_learning.py
python test_new_features_independent.py
python test_simple.py
```


Comparison with Original SIGnature
--------------------------------------------------------------------------------

| Feature | Original SIGnature | SIGnature_OOD |
|---------|-------------------|---------------|
| Gene Attribution | Integrated Gradients | Extended 4-level Attribution |
| Foundation Model Support | 4 models | Same |
| OOD Awareness | No | Structured OOD Detection |
| Representation Disentanglement | No | 3-space decomposition + G×E |
| Curriculum Learning | No | Progressive OOD Introduction |
| Transition Token | No | Program-level Decomposition |
| Calibrated Classifier | No | 3-level Classification |
| Confidence System | No | 4-level Trust System |
| Gene Classification | No | Quadrant Classification |
| Attribution Validation | No | Multi-method Validation |
| Backward Compatibility | - | Yes |


License
--------------------------------------------------------------------------------

MIT License


Citation
--------------------------------------------------------------------------------

To cite SIGnature in publications please use:

**Scoring gene importance by interpreting single-cell foundation models.**
*Maxwell P. Gold et al.*, Nature Biotechnology (2026).
https://www.nature.com/articles/s41587-026-03112-5
