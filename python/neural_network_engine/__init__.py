"""
Neural Network Engine - High-performance pure-Rust deep learning engine with Python bindings.
"""

try:
    from .neural_network_engine import *
except ImportError:
    # Development fallback: try loading directly from target/debug or target/release
    import ctypes
    import importlib.machinery
    import importlib.util
    import os
    import sys
    from pathlib import Path

    root = Path(__file__).parent.parent.parent
    candidates = [
        root / "target" / "release" / "libneural_network_engine.so",
        root / "target" / "debug" / "libneural_network_engine.so",
        root / "target" / "release" / "neural_network_engine.dll",
        root / "target" / "debug" / "neural_network_engine.dll",
        root / "target" / "release" / "libneural_network_engine.dylib",
        root / "target" / "debug" / "libneural_network_engine.dylib",
    ]
    loaded = False
    for path in candidates:
        if path.exists():
            loader = importlib.machinery.ExtensionFileLoader("neural_network_engine", str(path))
            spec = importlib.machinery.ModuleSpec("neural_network_engine", loader, origin=str(path))
            mod = importlib.util.module_from_spec(spec)
            loader.exec_module(mod)
            sys.modules["neural_network_engine.neural_network_engine"] = mod
            this_mod = sys.modules[__name__]
            for attr in dir(mod):
                if not attr.startswith("__"):
                    setattr(this_mod, attr, getattr(mod, attr))
            loaded = True
            break
    if not loaded:
        raise ImportError(
            "Could not import neural_network_engine extension module. "
            "Please build with `cargo build --features python` or install with `pip install .`"
        )
