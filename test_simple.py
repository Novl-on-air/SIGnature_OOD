import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from SIGnature import SIGnature
from SIGnature.config import get_default_config

config = get_default_config()
config.training.device = "cpu"
config.training.stage1.max_epochs = 2
config.training.stage2.max_epochs = 2
config.training.stage3.max_epochs = 2
config.training.stage4.max_epochs = 2

model = SIGnature(config=config)

model_path = os.path.join(os.path.dirname(__file__), "models", "scimilarity")
print(f"Model path: {model_path}")

model.load_foundation_model(model_type="scsimilarity", model_path=model_path)
print("Model loaded successfully")

print(f"Gene order: {len(model.gene_order)} genes")
print(f"Latent dim: {model.foundation_model.latent_dim}")

print("\n✅ Test passed!")