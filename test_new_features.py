import sys
import os
import torch
import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from SIGnature import SIGnature
from SIGnature.config import get_default_config
from SIGnature.representation.disentanglement import Disentangler
from SIGnature.transition.program_token import ProgramTokenBank
from SIGnature.attribution.attribution_validation import AttributionValidator
from SIGnature.attribution.gene_classification import GeneQuadrantClassifier

print("=" * 70)
print("Testing New Features")
print("=" * 70)

model_path = os.path.join(os.path.dirname(__file__), "models", "scimilarity")
config = get_default_config()
config.training.device = "cpu"
config.training.stage1.max_epochs = 3
config.training.stage2.max_epochs = 3
config.training.stage3.max_epochs = 3
config.training.stage4.max_epochs = 3

model = SIGnature(config=config)

print("\n1. Loading scSimilarity model...")
model.load_foundation_model(model_type="scsimilarity", model_path=model_path)
print(f"   ✓ Model loaded: {len(model.gene_order)} genes, {model.foundation_model.latent_dim} dim")

n_cells = 30
n_genes = len(model.gene_order)
X = csr_matrix(np.random.rand(n_cells, n_genes))

print("\n2. Testing Bilinear Layer (G×E Interaction)...")
try:
    z_geo = torch.randn(10, model.foundation_model.latent_dim)
    domain_labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    z_shared_u, z_shared_c, z_private = model.training_pipeline.disentangler(z_geo, domain_labels)
    print(f"   ✓ Bilinear layer works:")
    print(f"     - z_shared_u shape: {z_shared_u.shape}")
    print(f"     - z_shared_c shape: {z_shared_c.shape}")
    print(f"     - z_private shape: {z_private.shape}")
    print(f"     - z_shared_c != z_shared_u: {(z_shared_c != z_shared_u).any().item()}")
except Exception as e:
    print(f"   ✗ Bilinear layer failed: {e}")

print("\n3. Testing Token Biology Initialization...")
try:
    token_bank = model.training_pipeline.program_token_bank
    reference_data = {
        "oxidative_stress": torch.randn(model.config.model.transition.program_token.token_dim),
        "dna_damage": torch.randn(model.config.model.transition.program_token.token_dim),
    }
    token_bank.initialize_with_biology_reference(reference_data, gene_order=model.gene_order)
    print(f"   ✓ Token initialization with biology reference works")
    
    directions, anchors = token_bank.forward()
    print(f"   ✓ Token bank forward works: {directions.shape} directions")
except Exception as e:
    print(f"   ✗ Token initialization failed: {e}")

print("\n4. Testing Token Validation...")
try:
    validation_data = {
        "oxidative_stress": (torch.randn(5, model.config.model.transition.program_token.token_dim), 
                            torch.randn(5, model.config.model.transition.program_token.token_dim)),
    }
    scores = token_bank.validate_tokens(validation_data)
    report = token_bank.get_token_quality_report()
    print(f"   ✓ Token validation works")
    print(f"   ✓ Token quality report generated")
except Exception as e:
    print(f"   ✗ Token validation failed: {e}")

print("\n5. Testing Attribution Validation...")
try:
    validator = AttributionValidator(model.training_pipeline.geometry_adapter, model.gene_order)
    inputs = torch.randn(5, n_genes)
    baseline = torch.zeros(5, n_genes)
    attributions = torch.randn(5, n_genes)
    
    result = validator.validate_all(inputs, baseline, attributions)
    print(f"   ✓ Attribution validation works:")
    print(f"     - Overall confidence: {result['overall_confidence']:.4f}")
    print(f"     - Multipath consistency: {result['multipath']['consistency_score']:.4f}")
    print(f"     - Perturbation agreement: {result['perturbation']['agreement_score']:.4f}")
except Exception as e:
    print(f"   ✗ Attribution validation failed: {e}")

print("\n6. Testing Gene Quadrant Classification...")
try:
    classifier = GeneQuadrantClassifier(model.gene_order[:100])
    state_scores = torch.randn(100)
    transition_scores = torch.randn(100)
    
    classifications = classifier.classify_genes(state_scores, transition_scores)
    summary = classifier.get_classification_summary(classifications)
    
    print(f"   ✓ Gene classification works:")
    for gene_type, info in summary.items():
        print(f"     - {info['label']}: {info['count']} genes ({info['percentage']:.1f}%)")
    
    quadrant_scores = classifier.compute_quadrant_scores(state_scores, transition_scores)
    top_genes = classifier.get_top_genes_by_type(classifications, quadrant_scores)
    print(f"   ✓ Quadrant scores and top genes computed")
except Exception as e:
    print(f"   ✗ Gene classification failed: {e}")

print("\n7. Testing Full Pipeline Training...")
try:
    model.training_pipeline._fit_stage1(X)
    print("   ✓ Stage 1 (Geometry) training passed")
    
    model.training_pipeline._fit_stage2(X)
    print("   ✓ Stage 2 (Disentanglement) training passed")
    
    model.training_pipeline._fit_stage3(X)
    print("   ✓ Stage 3 (Transition) training passed")
    
    model.training_pipeline._fit_stage4(X)
    print("   ✓ Stage 4 (Classification) training passed")
except Exception as e:
    print(f"   ✗ Pipeline training failed: {e}")

print("\n8. Testing Inference...")
try:
    result = model.predict(X)
    print(f"   ✓ Inference passed")
    print(f"     - OOD predictions: {result.ood_predictions.shape}")
    print(f"     - Confidence levels: {list(result.confidence.keys())}")
except Exception as e:
    print(f"   ✗ Inference failed: {e}")

print("\n" + "=" * 70)
print("✅ All New Features Tested!")
print("=" * 70)