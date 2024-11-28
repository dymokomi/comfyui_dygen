import numpy as np
import torch, os
from PIL import Image, ImageDraw
import random
import folder_paths
from typing import Tuple

class AdaptiveColorLinesNode:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_lines": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100000,
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
                "alpha": ("INT", {
                    "default": 255,
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
    FUNCTION = "draw_adaptive_lines"
    CATEGORY = "DyGen/effects"

    def get_random_alpha(self, base_a: int, alpha_dev: int) -> int:
        return max(0, min(255, base_a + random.randint(-alpha_dev, alpha_dev)))

    def sample_color_at_point(self, img: Image.Image, x: int, y: int) -> Tuple[int, int, int]:
        # Ensure coordinates are within image bounds
        x = max(0, min(x, img.width - 1))
        y = max(0, min(y, img.height - 1))
        
        # Get color at point
        pixel = img.getpixel((x, y))
        if len(pixel) > 3:  # If RGBA, just take RGB components
            return pixel[:3]
        return pixel

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
        
        # Calculate center point
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        return ((x1, y1), (x2, y2), (center_x, center_y))

    def draw_adaptive_lines(self, image: torch.Tensor, num_lines: int, avg_thickness: float, 
                          thickness_deviation: float, avg_length: int, length_deviation: int,
                          alpha: int, alpha_deviation: int) :
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

        filename_prefix = self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, image.shape[1], image.shape[0])
        
        # Draw random lines
        for _ in range(num_lines):
            # Get random line parameters
            start_point, end_point, center_point = self.get_random_line_params(width, height, avg_length, length_deviation)
            
            # Sample color from center point
            sampled_color = self.sample_color_at_point(pil_image, center_point[0], center_point[1])
            
            # Get random alpha
            a = self.get_random_alpha(alpha, alpha_deviation)
            
            # Combine sampled color with alpha
            color = (*sampled_color, a)
            
            # Get random thickness
            thickness = max(0.1, avg_thickness + random.uniform(-thickness_deviation, thickness_deviation))
            
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

        filename_with_batch_num = filename.replace("%batch_num%", str(0))
        file = f"{filename_with_batch_num}_{counter:05}_.png"
        pil_image.save(os.path.join(full_output_folder, file))
        return {
            "ui": {
                "images": [
                    {
                        "filename": file,
                        "subfolder": subfolder,
                        "type": self.type
                    }
                ]
            },
            "result": (tensor_image,)
        }

NODE_CLASS_MAPPINGS = {
    "AdaptiveColorLines": AdaptiveColorLinesNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdaptiveColorLines": "DY Sampled Color Lines"
}