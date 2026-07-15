import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from SIGnature.transition.program_token import ProgramTokenBank, DEFAULT_TOKEN_CATEGORIES, DEFAULT_BIOLOGY_REFERENCE_GENES

print("=" * 70)
print("Testing Custom Token Biology Initialization")
print("=" * 70)

print("\n1. Using default tokens and genes...")
token_dim = 64
token_bank = ProgramTokenBank(token_dim)
print(f"   ✓ Default categories: {token_bank.get_all_categories()}")
print(f"   ✓ Total tokens: {token_bank.n_tokens}")
print(f"   ✓ Token names: {token_bank.get_all_token_names()[:5]}...")

print("\n2. Creating custom token categories...")
custom_categories = {
    "immune_response": ["t_cell_activation", "b_cell_proliferation", "cytokine_signaling", "macrophage_polarization"],
    "development": ["stem_cell_maintenance", "differentiation", "apoptosis_regulation", "cell_migration"],
}
custom_genes = {
    "t_cell_activation": ["CD3D", "CD3E", "CD3G", "CD28", "IL2", "IFNG", "TNF", "CD4"],
    "b_cell_proliferation": ["CD19", "CD20", "CD40", "BCL6", "MYC", "CXCR5", "CD79A", "CD79B"],
    "cytokine_signaling": ["IL6", "IL1B", "IL10", "IFNG", "TNF", "STAT3", "JAK1", "JAK2"],
    "macrophage_polarization": ["CD68", "CD163", "CD86", "CCL2", "IL12B", "ARG1", "MRC1", "TLR4"],
    "stem_cell_maintenance": ["OCT4", "SOX2", "NANOG", "KLF4", "MYC", "LIN28A", "ESRG", "DPPA4"],
    "differentiation": ["RUNX1", "GATA1", "PU1", "CEBPA", "ID1", "ID2", "TCF7", "LEF1"],
}
custom_token_bank = ProgramTokenBank(token_dim, token_categories=custom_categories, biology_reference_genes=custom_genes)
print(f"   ✓ Custom categories: {custom_token_bank.get_all_categories()}")
print(f"   ✓ Custom token names: {custom_token_bank.get_all_token_names()}")
print(f"   ✓ Reference genes for t_cell_activation: {custom_token_bank.biology_reference_genes['t_cell_activation']}")

print("\n3. Adding new token category...")
custom_token_bank.add_token_category(
    category_name="cancer",
    token_names=["cell_proliferation", "angiogenesis", "metastasis", "drug_resistance"],
    reference_genes={
        "cell_proliferation": ["PCNA", "KI67", "CCND1", "CDK4", "CDK6", "MYC", "BCL2", "MCM2"],
        "angiogenesis": ["VEGFA", "VEGFR2", "ANGPT1", "ANGPT2", "PDGF", "FGF2", "TIE2", "NOS3"],
    },
)
print(f"   ✓ After adding cancer category:")
print(f"     - Categories: {custom_token_bank.get_all_categories()}")
print(f"     - Total tokens: {custom_token_bank.n_tokens}")
print(f"     - Cancer tokens: {custom_token_bank.get_category_tokens('cancer')}")

print("\n4. Adding token to existing category...")
custom_token_bank.add_token_to_category(
    category_name="cancer",
    token_name="epithelial_mesenchymal_transition",
    reference_genes=["SNAI1", "SNAI2", "TWIST1", "ZEB1", "ZEB2", "VIM", "CDH1", "CDH2"],
)
print(f"   ✓ After adding EMT token:")
print(f"     - Cancer tokens: {custom_token_bank.get_category_tokens('cancer')}")
print(f"     - EMT reference genes: {custom_token_bank.biology_reference_genes['epithelial_mesenchymal_transition']}")

print("\n5. Setting reference genes for individual token...")
custom_token_bank.set_reference_genes(
    token_name="drug_resistance",
    genes=["MDR1", "MRP1", "BCRP", "CYP3A4", "ABCB1", "ABCC1", "ABCG2", "CYP2D6"],
)
print(f"   ✓ drug_resistance reference genes updated: {custom_token_bank.biology_reference_genes['drug_resistance']}")

print("\n6. Setting reference genes for entire category...")
custom_token_bank.set_reference_genes_for_category(
    category_name="development",
    genes_dict={
        "stem_cell_maintenance": ["POU5F1", "SOX2", "NANOG", "KLF4", "MYC", "LIN28A"],
        "differentiation": ["RUNX1", "GATA1", "PU1", "CEBPA", "ID1", "ID2"],
    },
)
print(f"   ✓ Development category genes updated")

print("\n7. Custom initialization function...")
gene_order = ["CD3D", "CD3E", "CD3G", "CD28", "IL2", "IFNG", "TNF", "CD4", "PCNA", "KI67"]

def my_custom_init(token_name, category, gene_order_list):
    genes = custom_token_bank.biology_reference_genes.get(token_name, [])
    if gene_order_list and len(genes) > 0:
        gene_indices = [gene_order_list.index(g) for g in genes if g in gene_order_list]
        if len(gene_indices) > 0:
            init_vector = torch.zeros(token_dim)
            init_vector[gene_indices[:token_dim]] = 1.0
            return init_vector / (init_vector.norm() + 1e-8)
    return torch.randn(token_dim) / (token_dim ** 0.5)

custom_token_bank.initialize_with_biology_reference(
    gene_order=gene_order,
    custom_init_fn=my_custom_init,
)
directions, anchors = custom_token_bank.forward()
print(f"   ✓ Custom initialization complete")
print(f"   ✓ Directions shape: {directions.shape}")
print(f"   ✓ Anchors shape: {anchors.shape}")

print("\n8. Initialization from gene expression data...")
gene_expression = {
    "t_cell_activation": torch.tensor([1.0, 2.0, 1.5, 0.8, 3.0, 2.5, 1.2, 0.5, 0.2, 0.1]),
    "cell_proliferation": torch.tensor([0.1, 0.2, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 3.0, 2.8]),
}
custom_token_bank.initialize_from_gene_expression(
    gene_expression=gene_expression,
    gene_order=gene_order,
)
print(f"   ✓ Gene expression initialization complete")

print("\n" + "=" * 70)
print("✅ Custom Token Biology Initialization Tested Successfully!")
print("=" * 70)