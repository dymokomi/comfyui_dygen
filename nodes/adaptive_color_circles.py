import numpy as np
import torch, os
from PIL import Image, ImageDraw
import random
import folder_paths
from typing import Tuple

class AdaptiveColorCirclesNode:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_circles": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100000,
                    "step": 1
                }),
                "min_radius": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 500,
                    "step": 1
                }),
                "max_radius": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "alpha": ("INT", {
                    "default": 128,
                    "min": 0,
                    "max": 255,
                    "step": 1
                }),
                "alpha_deviation": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 255,
                    "step": 1
                }),
                "outline_thickness": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "step": 1
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw_adaptive_circles"
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

    def get_random_circle_params(self, width: int, height: int, min_radius: int, max_radius: int) -> Tuple[Tuple[int, int], int]:
        # Get random center point
        x = random.randint(0, width)
        y = random.randint(0, height)
        
        # Get random radius
        radius = random.randint(min_radius, max_radius)
        
        return ((x, y), radius)

    def draw_adaptive_circles(self, image: torch.Tensor, num_circles: int, 
                            min_radius: int, max_radius: int,
                            alpha: int, alpha_deviation: int,
                            outline_thickness: int) -> torch.Tensor:
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
        
        # Draw random circles
        for _ in range(num_circles):
            # Get random circle parameters
            center_point, radius = self.get_random_circle_params(width, height, min_radius, max_radius)
            
            # Sample color from center point
            sampled_color = self.sample_color_at_point(pil_image, center_point[0], center_point[1])
            
            # Get random alpha
            a = self.get_random_alpha(alpha, alpha_deviation)
            
            # Combine sampled color with alpha
            color = (*sampled_color, a)
            
            # Calculate bounding box for circle
            bbox = [
                center_point[0] - radius,  # left
                center_point[1] - radius,  # top
                center_point[0] + radius,  # right
                center_point[1] + radius   # bottom
            ]
            
            # Draw filled circle
            draw.ellipse(bbox, fill=color)
            
            # Draw outline if thickness > 0
            if outline_thickness > 0:
                # Use same color but full opacity for outline
                outline_color = (*sampled_color, 255)
                draw.ellipse(bbox, fill=None, outline=outline_color, width=outline_thickness)
        
        # Composite the overlay onto the original image
        pil_image = Image.alpha_composite(pil_image, overlay)
        
        # Convert back to RGB for ComfyUI compatibility
        pil_image = pil_image.convert('RGB')
        
        # Convert back to tensor
        numpy_image = np.array(pil_image).astype(np.float32) / 255.0
        tensor_image = torch.from_numpy(numpy_image)
        
        # Add batch dimension
        tensor_image = tensor_image.unsqueeze(0)
        
        filename_prefix = self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, image.shape[1], image.shape[0])
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
    "AdaptiveColorCircles": AdaptiveColorCirclesNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdaptiveColorCircles": "Add Sampled Color Circles"
}