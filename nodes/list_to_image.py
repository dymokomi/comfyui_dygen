class ImageListToGrid:
    """Combines images from an IMAGE_LIST into a single grid IMAGE"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE_LIST",),
                "max_columns": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_grid"
    CATEGORY = "image/transform"
    
    def create_grid(self, images, max_columns=3):
        import torch
        
        if not images:
            raise ValueError("Empty image list provided")
            
        # Get dimensions from first image
        _, h, w = images[0].shape
        
        # Calculate grid dimensions
        num_images = len(images)
        num_columns = min(max_columns, num_images)
        num_rows = (num_images + num_columns - 1) // num_columns
        
        # Create rows first, then combine rows
        rows = []
        for row_idx in range(num_rows):
            # Get images for this row
            start_idx = row_idx * num_columns
            end_idx = min(start_idx + num_columns, num_images)
            row_images = images[start_idx:end_idx]
            
            # If this row is not full, pad with empty (black) images
            while len(row_images) < num_columns:
                row_images.append(torch.zeros_like(images[0]))
            
            # Concatenate images horizontally for this row
            row = torch.cat(row_images, dim=2)  # Concatenate along width
            rows.append(row)
        
        # Concatenate all rows vertically
        grid = torch.cat(rows, dim=1)  # Concatenate along height
        
        return (grid,)

NODE_CLASS_MAPPINGS = {
    "ImageListToGrid": ImageListToGrid
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageListToGrid": "Image List to Grid"
}