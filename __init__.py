from .nodes.quantize import DYImageQuantizeNode
from .nodes.cluster import DYImageClusterNode
from .nodes.palette import DYImagePaletteNode
from .nodes.masks import DYImageMasksNode
from .nodes.image_scaler import ImageScaler
from .nodes.list_to_image import ImageListToGrid
from .nodes.random_lines import RandomLinesNode
from .nodes.adaptive_color_lines import AdaptiveColorLinesNode
from .nodes.adaptive_color_circles import AdaptiveColorCirclesNode
from .nodes.adaptive_color_rectangles import AdaptiveColorRectanglesNode
from .nodes.binary_pattern_stamper import BinaryPatternStamperNode

NODE_CLASS_MAPPINGS = {
    "DYImageQuantize": DYImageQuantizeNode,
    "DYImageCluster": DYImageClusterNode,
    "DYImagePalette": DYImagePaletteNode,
    "DYImageMasks": DYImageMasksNode,
    "ImageListToGrid": ImageListToGrid,
    "ImageScaler": ImageScaler,
    "RandomLines": RandomLinesNode,
    "AdaptiveColorLines": AdaptiveColorLinesNode,
    "AdaptiveColorCircles": AdaptiveColorCirclesNode,
    "AdaptiveColorRectangles": AdaptiveColorRectanglesNode,
    "BinaryPatternStamper": BinaryPatternStamperNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DYImageQuantize": "DY Image Quantize",
    "DYImageCluster": "DY Image Cluster",
    "DYImagePalette": "DY Image Palette",
    "DYImageMasks": "DY Image Masks",
    "ImageListToGrid": "DY Image List to Grid",
    "ImageScaler": "DY Image Scaler",
    "RandomLines": "DY Random Lines",
    "AdaptiveColorLines": "DY Adaptive Color Lines",
    "AdaptiveColorCircles": "DY Adaptive Color Circles",
    "AdaptiveColorRectangles": "DY Adaptive Color Rectangles",
    "BinaryPatternStamper": "DY Binary Pattern Stamper"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']