import numpy as np
from PIL import Image
import torch

class ImageScaler:
    def __init__(self):
        self.type = "ImageScaler"
        self.output_node = True
        self.input_size_type = "INT"
        self.output_size_type = "INT"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "size": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "scale_image"
    CATEGORY = "DyGen/transform"

    def scale_image(self, image, size):
        # If image is a batch, process only the first image
        if len(image.shape) == 4:
            image = image[0]
            
        # Get current dimensions from tensor
        height, width = image.shape[:2]
        
        # Calculate new dimensions
        if width > height:
            new_width = size
            new_height = int(size * (height / width))
        else:
            new_height = size
            new_width = int(size * (width / height))
            
        # Convert from tensor to PIL Image
        pil_image = Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))
        
        # Resize the image
        resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert back to tensor and normalize to 0-1 range
        numpy_image = np.array(resized_image).astype(np.float32) / 255.0
        tensor_image = torch.from_numpy(numpy_image)
        
        # Add batch dimension
        tensor_image = tensor_image.unsqueeze(0)
            
        return (tensor_image,)

NODE_CLASS_MAPPINGS = {
    "ImageScaler": ImageScaler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageScaler": "Scale Image (Maintain Aspect)"
}