"""Prints the software/hardware versions used (for the manuscript) and a pinned requirements list.
Usage:  python print_env.py
"""
import importlib
import platform
import sys

print("Python        :", sys.version.split()[0])
print("OS            :", platform.platform())
print("Processor     :", platform.processor())
try:
    import psutil
    print("RAM (GB)      :", round(psutil.virtual_memory().total / 1e9, 1))
except Exception:
    print("RAM (GB)      : (pip install psutil to print it)")
print()
pins = []
for mod, pkg in [("torch", "torch"), ("torchvision", "torchvision"), ("ultralytics", "ultralytics"),
                 ("cv2", "opencv-python"), ("numpy", "numpy"), ("pandas", "pandas"), ("sklearn", "scikit-learn"),
                 ("PIL", "pillow"), ("matplotlib", "matplotlib"), ("seaborn", "seaborn")]:
    try:
        v = importlib.import_module(mod).__version__
        print(f"{pkg:14s}: {v}")
        pins.append(f"{pkg}=={v.split('+')[0]}")
    except Exception as e:
        print(f"{pkg:14s}: NOT FOUND ({e.__class__.__name__})")
try:
    import torch
    print("CUDA available:", torch.cuda.is_available(), "| torch threads:", torch.get_num_threads())
except Exception:
    pass
print("\n# requirements (pinned to this environment)")
print("\n".join(pins))
