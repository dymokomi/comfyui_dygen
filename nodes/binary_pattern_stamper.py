import numpy as np
import torch
from PIL import Image
import random
import os
from typing import Tuple
import time
import folder_paths

class BinaryPatternStamperNode:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        
    @classmethod
    def INPUT_TYPES(cls):
        patterns_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "patterns")
        
        # Create patterns directory if it doesn't exist
        if not os.path.exists(patterns_dir):
            os.makedirs(patterns_dir)
            
        # Get all PNG files from the patterns directory
        pattern_files = [f for f in os.listdir(patterns_dir) if f.lower().endswith('.png')]
        
        if not pattern_files:
            pattern_files = ["no_patterns_found.png"]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "pattern": (pattern_files,),
                "scale_multiplier": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 32,
                    "step": 1
                }),
                "stamp_probability": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "flip_h_probability": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "flip_v_probability": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "alpha": ("INT", {
                    "default": 255,
                    "min": 0,
                    "max": 255,
                    "step": 1
                }),
                "alpha_deviation": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 255,
                    "step": 1
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_binary_pattern"
    CATEGORY = "DyGen/effects"

    def get_random_alpha(self, base_a: int, alpha_dev: int) -> int:
        return max(0, min(255, base_a + random.randint(-alpha_dev, alpha_dev)))

    def sample_color_at_point(self, img: Image.Image, x: int, y: int) -> Tuple[int, int, int]:
        x = max(0, min(x, img.width - 1))
        y = max(0, min(y, img.height - 1))
        pixel = img.getpixel((x, y))
        if len(pixel) > 3:
            return pixel[:3]
        return pixel

    def apply_binary_pattern(self, image: torch.Tensor, pattern: str, 
                           scale_multiplier: int, stamp_probability: float,
                           flip_h_probability: float, flip_v_probability: float,
                           alpha: int, alpha_deviation: int, seed: int) -> torch.Tensor:
        # Set random seed
        if seed == -1:
            seed = int(time.time() * 1000) % (2**64)
        random.seed(seed)
        
        # Process image
        if len(image.shape) == 4:
            image = image[0]
        pil_image = Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')
            
        # Load and process base pattern
        patterns_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "patterns")
        pattern_path = os.path.join(patterns_dir, pattern)
        
        try:
            base_pattern = Image.open(pattern_path).convert('L')
        except:
            raise ValueError(f"Could not load pattern image from {pattern_path}")
        
        # Scale up pattern
        base_pattern = base_pattern.resize(
            (base_pattern.width * scale_multiplier, base_pattern.height * scale_multiplier), 
            Image.NEAREST
        )
        
        # Get dimensions
        pattern_width, pattern_height = base_pattern.size
        image_width, image_height = pil_image.size
        
        # Create overlay
        overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        
        # Calculate grid positions
        x_positions = range(0, image_width, pattern_width)
        y_positions = range(0, image_height, pattern_height)
        
        # Iterate over grid positions
        for y in y_positions:
            for x in x_positions:
                # Skip based on probability
                if random.random() > stamp_probability:
                    continue
                
                # Create a copy of the pattern for this stamp
                current_pattern = base_pattern.copy()
                
                # Apply random flips
                if random.random() < flip_h_probability:
                    current_pattern = current_pattern.transpose(Image.FLIP_LEFT_RIGHT)
                if random.random() < flip_v_probability:
                    current_pattern = current_pattern.transpose(Image.FLIP_TOP_BOTTOM)
                
                # Convert to array and create mask
                pattern_array = np.array(current_pattern)
                white_mask = pattern_array > 127
                
                # Get color from center
                center_x = x + pattern_width // 2
                center_y = y + pattern_height // 2
                if center_x >= image_width or center_y >= image_height:
                    continue
                    
                sampled_color = self.sample_color_at_point(pil_image, center_x, center_y)
                a = self.get_random_alpha(alpha, alpha_deviation)
                
                # Create stamp
                stamp_array = np.zeros((pattern_height, pattern_width, 4), dtype=np.uint8)
                stamp_array[white_mask] = (*sampled_color, a)
                stamp = Image.fromarray(stamp_array, 'RGBA')
                
                # Paste stamp
                overlay.paste(stamp, (x, y), stamp)
        
        # Composite and convert
        pil_image = Image.alpha_composite(pil_image, overlay)
        pil_image = pil_image.convert('RGB')
        
        # Return as tensor
        numpy_image = np.array(pil_image).astype(np.float32) / 255.0
        tensor_image = torch.from_numpy(numpy_image)
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
    "BinaryPatternStamper": BinaryPatternStamperNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BinaryPatternStamper": "Binary Pattern Stamper"
}