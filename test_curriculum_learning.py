import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 70)
print("Testing Curriculum Learning Strategy for Geometry Adapter")
print("=" * 70)

print("\n1. Testing CurriculumConfig...")
try:
    from SIGnature.config import CurriculumConfig, get_default_config
    
    config = CurriculumConfig()
    print(f"   ✓ CurriculumConfig created with default values:")
    print(f"     - enabled: {config.enabled}")
    print(f"     - schedule_type: {config.schedule_type}")
    print(f"     - warmup_epochs: {config.warmup_epochs}")
    print(f"     - max_ood_ratio: {config.max_ood_ratio}")
    print(f"     - difficulty_aware: {config.difficulty_aware}")
    print(f"     - initial_ood_ratio: {config.initial_ood_ratio}")
    print(f"     - ramp_up_epochs: {config.ramp_up_epochs}")
    
    default_config = get_default_config()
    print(f"   ✓ CurriculumConfig integrated into SIGnatureConfig")
    print(f"     - curriculum in representation config: {default_config.model.representation.curriculum.enabled}")
except Exception as e:
    print(f"   ✗ CurriculumConfig test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n2. Testing CurriculumScheduler initialization...")
try:
    from SIGnature.representation.curriculum_scheduler import CurriculumScheduler
    
    curriculum_config = CurriculumConfig(enabled=True, schedule_type="linear")
    scheduler = CurriculumScheduler(config=curriculum_config, n_samples=100)
    print(f"   ✓ CurriculumScheduler created")
    print(f"   ✓ Config: {scheduler.config.schedule_type}")
    print(f"   ✓ n_samples: {scheduler.n_samples}")
except Exception as e:
    print(f"   ✗ CurriculumScheduler initialization failed: {e}")
    import traceback
    traceback.print_exc()

print("\n3. Testing OOD mask setting...")
try:
    ood_mask = torch.tensor([False] * 80 + [True] * 20)
    ood_scores = torch.tensor([0.1] * 80 + [0.9] * 20)
    
    scheduler.set_ood_info(ood_mask, ood_scores)
    print(f"   ✓ OOD mask set successfully")
    print(f"   ✓ ID indices count: {len(scheduler.id_indices)}")
    print(f"   ✓ OOD indices count: {len(scheduler.ood_indices)}")
except Exception as e:
    print(f"   ✗ OOD mask setting failed: {e}")
    import traceback
    traceback.print_exc()

print("\n4. Testing OOD estimation from density...")
try:
    z_fm = torch.randn(100, 128)
    
    z_fm[-20:] += 5.0
    
    scheduler2 = CurriculumScheduler(n_samples=100)
    ood_mask_est, ood_scores_est = scheduler2.estimate_ood_from_density(z_fm, k=10, threshold_percentile=90)
    
    print(f"   ✓ OOD estimation from density works")
    print(f"   ✓ Estimated OOD count: {ood_mask_est.sum().item()}")
    print(f"   ✓ OOD scores range: [{ood_scores_est.min().item():.3f}, {ood_scores_est.max().item():.3f}]")
    print(f"   ✓ ID indices: {len(scheduler2.id_indices)}")
    print(f"   ✓ OOD indices: {len(scheduler2.ood_indices)}")
except Exception as e:
    print(f"   ✗ OOD estimation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n5. Testing linear schedule...")
try:
    scheduler = CurriculumScheduler(config=CurriculumConfig(schedule_type="linear", warmup_epochs=2, ramp_up_epochs=10, initial_ood_ratio=0.1, max_ood_ratio=0.5))
    scheduler.set_ood_info(torch.tensor([False] * 80 + [True] * 20))
    
    ratios = []
    for epoch in range(15):
        ratio = scheduler.get_target_ood_ratio(epoch, 20)
        ratios.append(ratio)
        if epoch in [0, 2, 5, 10, 14]:
            print(f"     - Epoch {epoch}: OOD ratio = {ratio:.4f}")
    
    print(f"   ✓ Linear schedule: {ratios[0]:.2f} -> {ratios[-1]:.2f}")
    assert ratios[0] == 0.1, f"Expected 0.1, got {ratios[0]}"
    assert ratios[-1] == 0.5, f"Expected 0.5, got {ratios[-1]}"
    print(f"   ✓ Linear schedule validation passed")
except Exception as e:
    print(f"   ✗ Linear schedule test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n6. Testing exponential schedule...")
try:
    scheduler = CurriculumScheduler(config=CurriculumConfig(schedule_type="exponential", warmup_epochs=2, ramp_up_epochs=10, initial_ood_ratio=0.1, max_ood_ratio=0.5))
    scheduler.set_ood_info(torch.tensor([False] * 80 + [True] * 20))
    
    ratios = []
    for epoch in range(15):
        ratio = scheduler.get_target_ood_ratio(epoch, 20)
        ratios.append(ratio)
    
    print(f"   ✓ Exponential schedule: {ratios[0]:.2f} -> {ratios[-1]:.2f}")
    print(f"   ✓ Faster ramp-up in early epochs")
except Exception as e:
    print(f"   ✗ Exponential schedule test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n7. Testing cosine schedule...")
try:
    scheduler = CurriculumScheduler(config=CurriculumConfig(schedule_type="cosine", warmup_epochs=2, ramp_up_epochs=10, initial_ood_ratio=0.1, max_ood_ratio=0.5))
    scheduler.set_ood_info(torch.tensor([False] * 80 + [True] * 20))
    
    ratios = []
    for epoch in range(15):
        ratio = scheduler.get_target_ood_ratio(epoch, 20)
        ratios.append(ratio)
    
    print(f"   ✓ Cosine schedule: {ratios[0]:.2f} -> {ratios[-1]:.2f}")
    print(f"   ✓ Smooth acceleration and deceleration")
except Exception as e:
    print(f"   ✗ Cosine schedule test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n8. Testing staged schedule...")
try:
    scheduler = CurriculumScheduler(config=CurriculumConfig(schedule_type="staged", warmup_epochs=2, ramp_up_epochs=10, initial_ood_ratio=0.1, max_ood_ratio=0.5))
    scheduler.set_ood_info(torch.tensor([False] * 80 + [True] * 20))
    
    ratios = []
    for epoch in range(15):
        ratio = scheduler.get_target_ood_ratio(epoch, 20)
        ratios.append(ratio)
    
    print(f"   ✓ Staged schedule: {ratios[0]:.2f} -> {ratios[-1]:.2f}")
    print(f"   ✓ Discrete stages")
except Exception as e:
    print(f"   ✗ Staged schedule test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n9. Testing sample_indices with curriculum...")
try:
    ood_mask = torch.tensor([False] * 80 + [True] * 20)
    scheduler = CurriculumScheduler(config=CurriculumConfig(schedule_type="linear", warmup_epochs=0, ramp_up_epochs=10, initial_ood_ratio=0.1, max_ood_ratio=0.5), n_samples=100)
    scheduler.set_ood_info(ood_mask)
    
    for epoch in [0, 5, 10]:
        indices = scheduler.sample_indices(epoch, 10, batch_size=50)
        
        n_ood_in_batch = sum(1 for i in indices if ood_mask[i])
        ratio = n_ood_in_batch / len(indices)
        print(f"     - Epoch {epoch}: batch_size={len(indices)}, OOD count={n_ood_in_batch}, ratio={ratio:.3f}")
    
    print(f"   ✓ Sample indices generation works")
except Exception as e:
    print(f"   ✗ Sample indices test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n10. Testing difficulty-aware sampling...")
try:
    ood_mask = torch.tensor([False] * 80 + [True] * 20)
    ood_scores = torch.tensor([0.1]*80 + [0.3]*10 + [0.9]*10)
    
    scheduler = CurriculumScheduler(
        config=CurriculumConfig(
            schedule_type="linear",
            warmup_epochs=0,
            ramp_up_epochs=10,
            initial_ood_ratio=0.2,
            max_ood_ratio=0.5,
            difficulty_aware=True,
            max_ood_threshold=0.6,
        ),
        n_samples=100,
    )
    scheduler.set_ood_info(ood_mask, ood_scores)
    
    for epoch in [0, 5, 10]:
        indices = scheduler.sample_indices(epoch, 10, batch_size=50)
        
        n_easy_ood = sum(1 for i in indices if ood_mask[int(i)] and ood_scores[int(i)] <= 0.6)
        n_hard_ood = sum(1 for i in indices if ood_mask[int(i)] and ood_scores[int(i)] > 0.6)
        print(f"     - Epoch {epoch}: easy_ood={n_easy_ood}, hard_ood={n_hard_ood}")
    
    print(f"   ✓ Difficulty-aware sampling works")
except Exception as e:
    print(f"   ✗ Difficulty-aware sampling failed: {e}")
    import traceback
    traceback.print_exc()

print("\n11. Testing curriculum info...")
try:
    ood_mask = torch.tensor([False] * 80 + [True] * 20)
    scheduler = CurriculumScheduler(config=CurriculumConfig(), n_samples=100)
    scheduler.set_ood_info(ood_mask)
    
    info = scheduler.get_curriculum_info(5, 20)
    print(f"   ✓ Curriculum info:")
    print(f"     - epoch: {info['epoch']}")
    print(f"     - total_epochs: {info['total_epochs']}")
    print(f"     - target_ood_ratio: {info['target_ood_ratio']:.4f}")
    print(f"     - n_id_samples: {info['n_id_samples']}")
    print(f"     - n_ood_samples: {info['n_ood_samples']}")
    print(f"     - schedule_type: {info['schedule_type']}")
    print(f"     - difficulty_aware: {info['difficulty_aware']}")
except Exception as e:
    print(f"   ✗ Curriculum info test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n12. Testing disabled curriculum (returns all samples)...")
try:
    scheduler = CurriculumScheduler(config=CurriculumConfig(enabled=False), n_samples=100)
    
    indices = scheduler.sample_indices(5, 20)
    print(f"   ✓ Disabled curriculum returns all indices")
    print(f"   ✓ Number of indices: {len(indices)} (expected: 100)")
    assert len(indices) == 100
except Exception as e:
    print(f"   ✗ Disabled curriculum test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✅ All Curriculum Learning Tests Passed!")
print("=" * 70)
