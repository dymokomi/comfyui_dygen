import torch
from PIL import Image, ImageFilter
import numpy as np

class DYImageClusterNode:
    """
    A node that clsuters images using PIL's Image Filter with configurable parameters.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "cluster_image"
    CATEGORY = "dygen/effects"

    def cluster_image(self, image, radius=16):
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

        # Quantize the image
        try:
            # Convert to RGB mode if not already (handles RGBA, L, etc.)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
                
            # Perform quantization
            clustered_image = pil_image.filter(ImageFilter.ModeFilter(size=radius))
            
            # Convert back to RGB mode
            output_image = clustered_image.convert('RGB')
            
        except Exception as e:
            raise RuntimeError(f"Cluster failed: {str(e)}")

        # Convert back to numpy array and then to tensor
        output_np = np.array(output_image).astype(np.float32) / 255.0
        output_tensor = torch.from_numpy(output_np).unsqueeze(0)

        return (output_tensor,)

    @classmethod
    def VALIDATE_INPUTS(cls, radius):
        if radius < 0 or radius > 100:
            return ["Radius must be between 0 and 100"]
        return True

# Node registration
NODE_CLASS_MAPPINGS = {
    "DYImageCluster": DYImageClusterNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DYImageCluster": "DY Image Cluster"
}