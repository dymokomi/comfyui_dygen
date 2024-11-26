import torch
from PIL import Image
import numpy as np

class DYImageQuantizeNode:
    """
    A node that quantizes images using PIL's quantize function with configurable parameters.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "colors": ("INT", {
                    "default": 256,
                    "min": 2,
                    "max": 256,
                    "step": 1
                }),
                "method": (["MEDIANCUT", "MAXCOVERAGE", "FASTOCTREE", "LIBIMAGEQUANT"],),
                "kmeans": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "quantize_image"
    CATEGORY = "dygen/effects"

    def quantize_image(self, image, colors=256, method="MEDIANCUT", kmeans=0):
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

        # Convert method string to PIL constant
        method_map = {
            "MEDIANCUT": 0,
            "MAXCOVERAGE": 1,
            "FASTOCTREE": 2,
            "LIBIMAGEQUANT": 3
        }
        
        # Quantize the image
        try:
            # Convert to RGB mode if not already (handles RGBA, L, etc.)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
                
            # Perform quantization
            quantized = pil_image.quantize(
                colors=colors,
                method=method_map[method],
                kmeans=kmeans
            )
            
            # Convert back to RGB mode
            output_image = quantized.convert('RGB')
            
        except Exception as e:
            raise RuntimeError(f"Quantization failed: {str(e)}")

        # Convert back to numpy array and then to tensor
        output_np = np.array(output_image).astype(np.float32) / 255.0
        output_tensor = torch.from_numpy(output_np).unsqueeze(0)

        return (output_tensor,)

    @classmethod
    def VALIDATE_INPUTS(cls, colors, method, kmeans):
        if colors < 2 or colors > 256:
            return ["Number of colors must be between 2 and 256"]
        if kmeans < 0 or kmeans > 100:
            return ["K-means iterations must be between 0 and 100"]
        return True

# Node registration
NODE_CLASS_MAPPINGS = {
    "DYImageQuantize": DYImageQuantizeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DYImageQuantize": "DY Image Quantize"
}