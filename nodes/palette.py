import torch
from PIL import Image
import numpy as np
from typing import List, Tuple

class DYColorPaletteOutput:
    """
    Helper class for handling COLOR_PALETTE type in other nodes.
    A COLOR_PALETTE is a list of RGB tuples: [(r,g,b), ...]
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "palette": ("COLOR_PALETTE",),
            }
        }
    
class DYImagePaletteNode:
    """
    A node that clsuters images using PIL's Image Filter with configurable parameters.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("COLOR_PALETTE",)
    FUNCTION = "extract_colors"
    CATEGORY = "DyGen/generation"

    def extract_colors(self, image: torch.Tensor) -> Tuple[List[Tuple[int, int, int]], ...]:
        # Convert from tensor format [B, H, W, C] to numpy array
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if len(image_np.shape) == 4:
                image_np = image_np[0]  # Take first image if batched
        else:
            image_np = image

        # Convert numpy array (0-1 float) to PIL Image (0-255 uint8)
        image_np = (image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        # Convert to RGB if not already
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        try:
            # Get colors with their counts
            colors = pil_image.getcolors(maxcolors=256)
            
            if colors is None:
                raise ValueError(f"Image has more than {max_colors} colors. Increase max_colors parameter.")
            
            # Extract just the RGB values, discarding the counts
            rgb_colors = [x[1] for x in colors]
            
            # Sort colors by their counts (most frequent first)
            sorted_colors = [x[1] for x in sorted(colors, key=lambda x: x[0], reverse=True)]
            
        except Exception as e:
            raise RuntimeError(f"Color extraction failed: {str(e)}")

        # Return as a tuple containing the list (ComfyUI expects a tuple)
        return (sorted_colors,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "DYImagePalette": DYImagePaletteNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DYImagePalette": "DY Image Palette"
}