#!/usr/bin/env python3
"""Quick environment check for HPC setup."""

import importlib.util
import sys


def check_module(name):
    spec = importlib.util.find_spec(name)
    status = "OK" if spec else "MISSING"
    print(f"  {name}: {status}")
    return spec is not None


def main():
    print("=== Python ===")
    print(f"  {sys.executable}")
    print(f"  {sys.version}")

    print("\n=== PyTorch ===")
    try:
        import torch
        print(f"  torch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("  MISSING - torch not installed!")
        sys.exit(1)

    print("\n=== Required packages ===")
    all_ok = True
    for m in ["isaacsim", "isaaclab", "gymnasium", "tensorboard", "numpy", "SwarmACB_isaac"]:
        if not check_module(m):
            all_ok = False

    if not all_ok:
        print("\nSome packages are MISSING!")
        sys.exit(1)

    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
