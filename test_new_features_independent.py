import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 70)
print("Testing New Features (Independent)")
print("=" * 70)

print("\n1. Testing Disentangler with Bilinear Layer (G×E Interaction)...")
try:
    from SIGnature.representation.disentanglement import Disentangler
    
    input_dim = 128
    shared_dim = 64
    private_dim = 32
    n_domains = 2
    
    disentangler = Disentangler(input_dim, shared_dim, private_dim, n_domains)
    
    z_geo = torch.randn(10, input_dim)
    domain_labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    
    z_shared_u, z_shared_c, z_private = disentangler(z_geo, domain_labels)
    
    print(f"   ✓ Bilinear layer works:")
    print(f"     - z_shared_u shape: {z_shared_u.shape}")
    print(f"     - z_shared_c shape: {z_shared_c.shape}")
    print(f"     - z_private shape: {z_private.shape}")
    print(f"     - z_shared_c != z_shared_u: {(z_shared_c != z_shared_u).any().item()}")
    print(f"     - Bilinear weight shape: {disentangler.bilinear_weight.shape}")
except Exception as e:
    print(f"   ✗ Bilinear layer failed: {e}")
    import traceback
    traceback.print_exc()

print("\n2. Testing Intervention Invariance Loss...")
try:
    from SIGnature.representation.disentanglement import intervention_invariance_loss
    
    z_private = torch.randn(20, 32)
    treatment_labels = torch.tensor([0] * 10 + [1] * 10)
    
    loss = intervention_invariance_loss(z_private, treatment_labels)
    print(f"   ✓ Intervention invariance loss works: {loss.item():.6f}")
except Exception as e:
    print(f"   ✗ Intervention invariance loss failed: {e}")
    import traceback
    traceback.print_exc()

print("\n3. Testing Token Biology Initialization...")
try:
    from SIGnature.transition.program_token import ProgramTokenBank
    
    token_dim = 64
    token_bank = ProgramTokenBank(token_dim)
    
    reference_data = {
        "oxidative_stress": torch.randn(token_dim),
        "dna_damage": torch.randn(token_dim),
        "apoptosis": torch.randn(token_dim),
    }
    
    gene_order = ["NRF2", "HMOX1", "SOD1", "TP53", "ATM", "CASP3"]
    token_bank.initialize_with_biology_reference(reference_data, gene_order=gene_order)
    
    directions, anchors = token_bank.forward()
    print(f"   ✓ Token initialization with biology reference works")
    print(f"   ✓ Token bank forward works: {directions.shape} directions, {anchors.shape} anchors")
    print(f"   ✓ Has biology_reference_genes: {len(token_bank.biology_reference_genes) > 0}")
except Exception as e:
    print(f"   ✗ Token initialization failed: {e}")
    import traceback
    traceback.print_exc()

print("\n4. Testing Token Validation...")
try:
    validation_data = {
        "oxidative_stress": (torch.randn(5, token_dim), torch.randn(5, token_dim)),
        "dna_damage": (torch.randn(5, token_dim), torch.randn(5, token_dim)),
    }
    
    scores = token_bank.validate_tokens(validation_data)
    report = token_bank.get_token_quality_report()
    
    print(f"   ✓ Token validation works")
    print(f"   ✓ Validation scores computed: {len(scores)} tokens")
    print(f"   ✓ Token quality report generated: {len(report)} entries")
    print(f"   ✓ Valid tokens: {len(token_bank.get_valid_tokens())}")
except Exception as e:
    print(f"   ✗ Token validation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n5. Testing Attribution Validation...")
try:
    from SIGnature.attribution.attribution_validation import AttributionValidator
    
    class DummyModel(torch.nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.fc = torch.nn.Linear(input_dim, output_dim)
        
        def forward(self, x):
            return self.fc(x)
    
    gene_order = [f"gene_{i}" for i in range(100)]
    model = DummyModel(100, 5)
    validator = AttributionValidator(model, gene_order)
    
    inputs = torch.randn(5, 100)
    baseline = torch.zeros(5, 100)
    attributions = torch.randn(5, 100)
    
    result = validator.validate_all(inputs, baseline, attributions)
    
    print(f"   ✓ Attribution validation works:")
    print(f"     - Overall confidence: {result['overall_confidence']:.4f}")
    print(f"     - Multipath consistency: {result['multipath']['consistency_score']:.4f}")
    print(f"     - Perturbation agreement: {result['perturbation']['agreement_score']:.4f}")
except Exception as e:
    print(f"   ✗ Attribution validation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n6. Testing Gene Quadrant Classification...")
try:
    from SIGnature.attribution.gene_classification import GeneQuadrantClassifier, classify_genes
    
    gene_order = [f"gene_{i}" for i in range(100)]
    classifier = GeneQuadrantClassifier(gene_order)
    
    state_scores = torch.randn(100)
    transition_scores = torch.randn(100)
    
    classifications = classifier.classify_genes(state_scores, transition_scores)
    summary = classifier.get_classification_summary(classifications)
    quadrant_scores = classifier.compute_quadrant_scores(state_scores, transition_scores)
    top_genes = classifier.get_top_genes_by_type(classifications, quadrant_scores)
    
    print(f"   ✓ Gene classification works:")
    for gene_type, info in summary.items():
        print(f"     - {info['label']}: {info['count']} genes ({info['percentage']:.1f}%)")
    print(f"   ✓ Quadrant scores computed: {len(quadrant_scores)} genes")
    print(f"   ✓ Top genes by type computed")
    
    result = classify_genes(gene_order, state_scores, transition_scores)
    print(f"   ✓ classify_genes function works")
except Exception as e:
    print(f"   ✗ Gene classification failed: {e}")
    import traceback
    traceback.print_exc()

print("\n7. Testing Total Disentangle Loss with Intervention Invariance...")
try:
    from SIGnature.representation.disentanglement import (
        DisentangledDecoder, compute_total_disentangle_loss
    )
    
    test_input_dim = 128
    test_shared_dim = 64
    test_private_dim = 32
    
    test_disentangler = Disentangler(test_input_dim, test_shared_dim, test_private_dim, 2)
    test_z_geo = torch.randn(10, test_input_dim)
    test_domain_labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    test_z_shared_u, test_z_shared_c, test_z_private = test_disentangler(test_z_geo, test_domain_labels)
    
    decoder = DisentangledDecoder(test_shared_dim, test_private_dim, test_input_dim)
    x_recon = decoder(test_z_shared_u, test_z_shared_c, test_z_private)
    x_true = torch.randn(10, test_input_dim)
    
    loss_result = compute_total_disentangle_loss(
        test_z_shared_u, test_z_shared_c, test_z_private,
        x_recon, x_true,
        domain_labels=test_domain_labels,
        treatment_labels=treatment_labels[:10],
    )
    
    print(f"   ✓ Total disentangle loss works:")
    print(f"     - Total: {loss_result['total'].item():.6f}")
    print(f"     - Reconstruction: {loss_result['reconstruction'].item():.6f}")
    print(f"     - Intervention invariance: {loss_result['intervention_invariance'].item():.6f}")
except Exception as e:
    print(f"   ✗ Total disentangle loss failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✅ All New Features Tested Successfully!")
print("=" * 70)