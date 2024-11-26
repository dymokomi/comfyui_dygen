from .nodes.quantize import DYImageQuantizeNode
from .nodes.cluster import DYImageClusterNode
from .nodes.palette import DYImagePaletteNode
from .nodes.masks import DYImageMasksNode

from .nodes.list_to_image import ImageListToGrid

NODE_CLASS_MAPPINGS = {
    "DYImageQuantize": DYImageQuantizeNode,
    "DYImageCluster": DYImageClusterNode,
    "DYImagePalette": DYImagePaletteNode,
    "DYImageMasks": DYImageMasksNode,
    "ImageListToGrid": ImageListToGrid
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DYImageQuantize": "DY Image Quantize",
    "DYImageCluster": "DY Image Cluster",
    "DYImagePalette": "DY Image Palette",
    "DYImageMasks": "DY Image Masks",
    "ImageListToGrid": "Image List to Grid"
}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']