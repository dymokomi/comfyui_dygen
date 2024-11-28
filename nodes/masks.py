import torch
from PIL import Image
import numpy as np

class DYImageMasksNode:
    """
    A node that creates masks from images using PIL's quantize function with configurable parameters.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "palette": ("COLOR_PALETTE",),
            },
        }

    RETURN_TYPES = ("IMAGE_LIST",)
    FUNCTION = "create_masks"
    CATEGORY = "DyGen/generation"

    def create_masks(self, image, palette):
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
                
            masks = []
            source_pixels = pil_image.load()
            for color in palette:
                img = Image.new("L", pil_image.size)
                pixels = img.load()
                for x in range(img.size[0]):
                    for y in range(img.size[1]): 
                        if source_pixels[x,y] == color:
                            pixels[x,y] = 255
                        else:
                            pixels[x,y] = 0
                masks.append(img)
            # Convert back to RGB mode
            output_masks = masks
            
        except Exception as e:
            raise RuntimeError(f"Quantization failed: {str(e)}")

        # Convert back to numpy array and then to tensor
        output_masks_tensor = []
        for mask in output_masks:
            output_np = np.array(mask).astype(np.float32) / 255.0
            output_tensor = torch.from_numpy(output_np).unsqueeze(0)
            output_masks_tensor.append(output_tensor)

        return (output_masks_tensor,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "DYImageMasks": DYImageMasksNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DYImageMasks": "DY Image Masks"
}