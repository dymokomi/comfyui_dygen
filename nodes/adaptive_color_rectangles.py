import numpy as np
import torch, os
from PIL import Image, ImageDraw
import random
from typing import Tuple
import folder_paths

class AdaptiveColorRectanglesNode:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_rectangles": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100000,
                    "step": 1
                }),
                "min_size": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 500,
                    "step": 1
                }),
                "max_size": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "aspect_ratio": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
                "aspect_variance": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1
                }),
                "rotation": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 180,
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
    FUNCTION = "draw_adaptive_rectangles"
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

    def get_random_rectangle_params(self, width: int, height: int, min_size: int, max_size: int, 
                                  aspect_ratio: float, aspect_variance: float, rotation: int) -> Tuple[list, Tuple[int, int]]:
        # Get random center point
        center_x = random.randint(0, width)
        center_y = random.randint(0, height)
        
        # Get random base size
        base_size = random.randint(min_size, max_size)
        
        # Get random aspect ratio variation
        actual_aspect = max(0.1, aspect_ratio + random.uniform(-aspect_variance, aspect_variance))
        
        # Calculate width and height
        rect_width = base_size
        rect_height = int(base_size / actual_aspect)
        
        # Calculate rectangle corners
        half_width = rect_width // 2
        half_height = rect_height // 2
        
        # Create corner points
        points = [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height)
        ]
        
        # Apply rotation if specified
        if rotation > 0:
            angle = random.uniform(-rotation, rotation)
            radian = np.radians(angle)
            cos_a = np.cos(radian)
            sin_a = np.sin(radian)
            points = [(int(x * cos_a - y * sin_a), int(x * sin_a + y * cos_a)) for x, y in points]
        
        # Translate to center position
        points = [(x + center_x, y + center_y) for x, y in points]
        
        return points, (center_x, center_y)

    def draw_adaptive_rectangles(self, image: torch.Tensor, num_rectangles: int, 
                               min_size: int, max_size: int,
                               aspect_ratio: float, aspect_variance: float,
                               rotation: int, alpha: int, alpha_deviation: int,
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
        
        # Draw random rectangles
        for _ in range(num_rectangles):
            # Get random rectangle parameters
            points, center = self.get_random_rectangle_params(
                width, height, min_size, max_size, 
                aspect_ratio, aspect_variance, rotation
            )
            
            # Sample color from center point
            sampled_color = self.sample_color_at_point(pil_image, center[0], center[1])
            
            # Get random alpha
            a = self.get_random_alpha(alpha, alpha_deviation)
            
            # Combine sampled color with alpha
            color = (*sampled_color, a)
            
            # Draw filled rectangle
            draw.polygon(points, fill=color)
            
            # Draw outline if thickness > 0
            if outline_thickness > 0:
                # Use same color but full opacity for outline
                outline_color = (*sampled_color, 255)
                draw.line(points + [points[0]], fill=outline_color, width=outline_thickness)
        
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
    "AdaptiveColorRectangles": AdaptiveColorRectanglesNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdaptiveColorRectangles": "Add Sampled Color Rectangles"
}