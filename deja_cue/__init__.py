"""Deja Cue reproduction package."""

# Keep package import lightweight. Callers import data and scan APIs from their
# defining modules, so inspecting metadata does not initialize PyTorch.
__all__ = ["__version__"]

__version__ = "1.0.0"
