import numpy as np
import torch
from PIL import Image, ImageDraw
import random
from typing import Tuple

class RandomLinesNode:
    def __init__(self):
        self.type = "RandomLines"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_lines": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "avg_thickness": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 0.1
                }),
                "thickness_deviation": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.1
                }),
                "avg_length": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 2000,
                    "step": 1
                }),
                "length_deviation": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 1000,
                    "step": 1
                }),
                "red": ("INT", {
                    "default": 255,
                    "min": 0,
                    "max": 255,
                    "step": 1
                }),
                "green": ("INT", {
                    "default": 255,
                    "min": 0,
                    "max": 255,
                    "step": 1
                }),
                "blue": ("INT", {
                    "default": 255,
                    "min": 0,
                    "max": 255,
                    "step": 1
                }),
                "alpha": ("INT", {
                    "default": 255,
                    "min": 0,
                    "max": 255,
                    "step": 1
                }),
                "color_deviation": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 255,
                    "step": 1
                }),
                "alpha_deviation": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 255,
                    "step": 1
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw_random_lines"
    CATEGORY = "DyGen/effects"

    def get_random_color(self, base_r: int, base_g: int, base_b: int, base_a: int, 
                        color_dev: int, alpha_dev: int) -> Tuple[int, int, int, int]:
        def clamp(value: int) -> int:
            return max(0, min(255, value))
        
        r = clamp(base_r + random.randint(-color_dev, color_dev))
        g = clamp(base_g + random.randint(-color_dev, color_dev))
        b = clamp(base_b + random.randint(-color_dev, color_dev))
        a = clamp(base_a + random.randint(-alpha_dev, alpha_dev))
        return (r, g, b, a)

    def get_random_line_params(self, width: int, height: int, avg_length: int, length_dev: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        # Get random starting point
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        
        # Get random angle in radians
        angle = random.uniform(0, 2 * np.pi)
        
        # Get random length
        length = max(1, avg_length + random.randint(-length_dev, length_dev))
        
        # Calculate end point
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        
        return ((x1, y1), (x2, y2))

    def draw_random_lines(self, image: torch.Tensor, num_lines: int, avg_thickness: float, 
                         thickness_deviation: float, avg_length: int, length_deviation: int,
                         red: int, green: int, blue: int, alpha: int,
                         color_deviation: int, alpha_deviation: int) -> torch.Tensor:
        # If image is batched, process only the first image
        if len(image.shape) == 4:
            image = image[0]
        
        # Convert tensor to PIL Image
        pil_image = Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))
        
        # Convert to RGBA if not already
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')
            
        width, height = pil_image.size
        
        # Create a transparent overlay
        overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Draw random lines
        for _ in range(num_lines):
            # Get random color with alpha
            color = self.get_random_color(red, green, blue, alpha, color_deviation, alpha_deviation)
            
            # Get random thickness
            thickness = max(0.1, avg_thickness + random.uniform(-thickness_deviation, thickness_deviation))
            
            # Get random line coordinates
            start_point, end_point = self.get_random_line_params(width, height, avg_length, length_deviation)
            
            # Draw the line
            draw.line([start_point, end_point], fill=color, width=int(thickness))
        
        # Composite the overlay onto the original image
        pil_image = Image.alpha_composite(pil_image, overlay)
        
        # Convert back to RGB for ComfyUI compatibility
        pil_image = pil_image.convert('RGB')
        
        # Convert back to tensor
        numpy_image = np.array(pil_image).astype(np.float32) / 255.0
        tensor_image = torch.from_numpy(numpy_image)
        
        # Add batch dimension
        tensor_image = tensor_image.unsqueeze(0)
        
        return (tensor_image,)

NODE_CLASS_MAPPINGS = {
    "RandomLines": RandomLinesNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomLines": "Add Random Lines"
}