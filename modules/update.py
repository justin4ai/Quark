import torch
import torch.nn.functional as F

class RayEncoder(torch.nn.Module):
    def __init__(self, encoding_dim = 16):

        super(RayEncoder, self).__init__()
        self.encoding_dim = encoding_dim

    def forward(self, ray_near, ray_far):

        ray_diff = ray_far - ray_near  
        ray_diff_normalized = torch.tanh(ray_diff)  
        encoded_rays = self.sinusoidal_encoding(ray_diff_normalized)  
        
        return encoded_rays

    def sinusoidal_encoding(self, inputs):

        freqs = 2 ** torch.arange(self.encoding_dim, dtype=inputs.dtype, device=inputs.device)  # Shape: (encoding_dim,)

        inputs_expanded = inputs.unsqueeze(-1) * freqs  
        sin_encoding = torch.sin(inputs_expanded) 
        cos_encoding = torch.cos(inputs_expanded)  
        
        encoding = torch.cat([sin_encoding, cos_encoding], dim = -1)  
        return encoding.view(encoding.shape[0], -1) 

class UpdateBlock():
    
    def concat_features(self, rendered_features, input_images, ray_directions):
        """
        Combines rendered features, input images, and ray encodings.

        Args:
            rendered_features: [B, M, H, W, C] Rendered features
            input_images: [B, M, H, W, 3] Input images
            ray_directions: [B, M, H, W, 3] Ray direction encodings

        Returns:
            Combined features: [B, M, H, W, C]
        """
        ray_features = torch.cat([input_images, ray_directions], dim = -1) 
        combined = torch.cat([rendered_features, ray_features], dim =-1) 
        return combined.permute(0, 4, 1, 2, 3).contiguous()