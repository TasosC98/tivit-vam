#!/usr/bin/env python3
"""Comprehensive debug script to identify device placement issues.

Key checks:
1. GPU availability
2. Model architecture before/after .to(device)
3. Lazy encoder initialization (the root cause!)
4. Forward pass with device tracking
"""

import sys
import torch
import torch.nn as nn
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)

def main():
    """Main debug routine."""
    
    # Step 1: Check Python and PyTorch
    print("\n" + "="*80)
    print("STEP 1: Environment Check")
    print("="*80)
    print(f"✓ Python version: {sys.version}")
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("✗ NO GPU DETECTED!")
        return
    
    # Step 2: Load config and build model
    print("\n" + "="*80)
    print("STEP 2: Model Loading")
    print("="*80)
    
    try:
        from tivit.core.config import load_experiment_config
        from tivit.models.factory import build_model
        
        config_path = Path("configs/default.yaml")
        if not config_path.exists():
            LOGGER.error(f"Config not found: {config_path}")
            return
        
        print(f"Loading config from: {config_path.resolve()}")
        cfg = dict(load_experiment_config([str(config_path)]))
        print("✓ Config loaded successfully")
        
    except Exception as e:
        LOGGER.error(f"Failed to load config: {e}")
        import traceback
        traceback.print_exc()
        return
    
    try:
        print("Building model...")
        model = build_model(cfg)
        print(f"✓ Model built: {type(model).__name__}")
        
    except Exception as e:
        LOGGER.error(f"Failed to build model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Inspect encoder before .to(device)
    print("\n" + "="*80)
    print("STEP 3: Lazy Encoder Detection (BEFORE .to(device))")
    print("="*80)
    
    has_encoder_before = hasattr(model, 'encoder') and model.encoder is not None
    print(f"Model has 'encoder' attribute: {hasattr(model, 'encoder')}")
    print(f"Encoder is initialized: {has_encoder_before}")
    
    if not has_encoder_before:
        print("⚠️  ISSUE FOUND: Encoder is LAZY-INITIALIZED!")
        print("    The encoder will be created during first forward pass on CPU!")
    
    check_model_devices(model, "BEFORE .to(device)")
    
    # Step 4: Move to device
    print("\n" + "="*80)
    print("STEP 4: Device Transfer")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Moving model to: {device}")
    
    model = model.to(device)
    print("✓ .to(device) called")
    
    # Re-check after moving
    check_model_devices(model, "AFTER .to(device)")
    
    if not hasattr(model, 'encoder') or model.encoder is None:
        print("⚠️  CRITICAL: Encoder still not initialized after .to(device)!")
    
    # Step 5: Force encoder initialization
    print("\n" + "="*80)
    print("STEP 5: Force Encoder Initialization")
    print("="*80)
    
    print("Calling _init_encoder_if_needed to force initialization...")
    try:
        model._init_encoder_if_needed(t_tokens=12, s_tokens=1536)
        print("✓ Encoder initialized")
        
        # Move encoder to device if it exists
        if model.encoder is not None:
            print(f"Moving encoder to {device}...")
            model.encoder = model.encoder.to(device)
            print("✓ Encoder moved to device")
        else:
            print("✗ Encoder still None after _init_encoder_if_needed!")
    
    except Exception as e:
        LOGGER.error(f"Failed to force encoder init: {e}")
        import traceback
        traceback.print_exc()
    
    check_model_devices(model, "AFTER encoder init and to(device)")
    
    # Step 6: Test forward pass
    print("\n" + "="*80)
    print("STEP 6: Forward Pass Test")
    print("="*80)
    
    try:
        batch_size = 1
        frames = 96
        channels = 3
        height = 180
        width = 512
        
        # IMPORTANT: Model expects (B, T, C, H, W) format, NOT (B, C, T, H, W)!
        print(f"Creating dummy input: (B={batch_size}, T={frames}, C={channels}, H={height}, W={width})")
        x = torch.randn(batch_size, frames, channels, height, width, device=device)
        print(f"✓ Input tensor created: shape={x.shape}, device={x.device}")
        
        print("Running forward pass (no_grad mode)...")
        model.eval()
        with torch.no_grad():
            output = model(x, return_per_tile=False)
        
        print("✓ Forward pass successful!")
        print("\nOutput tensors:")
        for key, val in output.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key:20s}: shape={str(val.shape):20s} device={val.device}")
        
        return True
    
    except Exception as e:
        LOGGER.error(f"✗ Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_model_devices(model, stage_name):
    """Detailed device placement inspection."""
    
    print(f"\n--- {stage_name} ---")
    
    param_devices = {}
    buffer_devices = {}
    
    # Collect parameter devices
    for name, param in model.named_parameters():
        dev = str(param.device)
        if dev not in param_devices:
            param_devices[dev] = []
        param_devices[dev].append(name)
    
    # Collect buffer devices
    for name, buf in model.named_buffers():
        dev = str(buf.device)
        if dev not in buffer_devices:
            buffer_devices[dev] = []
        buffer_devices[dev].append(name)
    
    # Print parameters
    print(f"\nParameters ({sum(len(v) for v in param_devices.values())} total):")
    for dev in sorted(param_devices.keys()):
        names = param_devices[dev]
        indicator = "✓" if "cuda" in dev else "✗"
        print(f"  {indicator} {dev}: {len(names)} parameters")
        if dev != 'cuda:0' and len(names) > 0:
            for name in names[:3]:
                print(f"      - {name}")
    
    # Print buffers
    print(f"\nBuffers ({sum(len(v) for v in buffer_devices.values())} total):")
    for dev in sorted(buffer_devices.keys()):
        names = buffer_devices[dev]
        indicator = "✓" if "cuda" in dev else "✗"
        print(f"  {indicator} {dev}: {len(names)} buffers")
        if dev != 'cuda:0' and len(names) > 0:
            for name in names[:3]:
                print(f"      - {name}")
    
    # Check for encoder
    if hasattr(model, 'encoder'):
        if model.encoder is not None:
            encoder_params = sum(1 for _ in model.encoder.parameters())
            encoder_dev = next(model.encoder.parameters()).device
            print(f"\nEncoder: {encoder_params} parameters, device={encoder_dev}")
        else:
            print(f"\nEncoder: NOT INITIALIZED (lazy init)")

if __name__ == "__main__":
    success = main()
    
    print("\n" + "="*80)
    print("DEBUG SUMMARY")
    print("="*80)
    if success:
        print("✓ All tests passed! Forward pass works correctly.")
    else:
        print("✗ Forward pass failed. See errors above.")
    print("="*80 + "\n")
