"""
Causal Reasoning Package for sPixel_CXGNN
ITE attribution (Integrated Gradients) and optional superpixel visualization.
"""

from .attribution import ICAttributor
from .visualizer import AttributionVisualizer

__version__ = "0.1.0"
__all__ = ["ICAttributor", "AttributionVisualizer"]
