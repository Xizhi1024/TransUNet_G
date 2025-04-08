# --- Add this block to a new file (e.g., frequency_module.py) or within vit_seg_modeling.py ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBnRelu(nn.Sequential):
    """Basic Conv-BN-ReLU block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not use_batchnorm)
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        super(ConvBnRelu, self).__init__(conv, bn, relu)

class FrequencySupplementBlock(nn.Module):
    """
    Extracts frequency features from the input image using FFT and a small CNN.
    """
    def __init__(self, in_channels=3, embed_dim=768, patch_size=16, target_grid_size=14, light_weight=True):
        """
        Args:
            in_channels (int): Number of input image channels (usually 1 for grayscale, 3 for RGB).
            embed_dim (int): Target embedding dimension for frequency features, should be compatible
                             with the spatial features for fusion (e.g., ViT hidden_size).
            patch_size (int): The patch size used by ViT to determine the target grid size.
            target_grid_size (int): Explicit target spatial grid size (H/patch_size, W/patch_size).
                                     Calculated if not provided based on a standard 224 input.
            light_weight (bool): If True, use fewer channels in intermediate layers.
        """
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        # Example assumes 224x224 input if target_grid_size not given explicitly
        self.target_grid_hw = target_grid_size # e.g., 224 // 16 = 14

        # Determine intermediate channels based on lightweight flag
        c1, c2, c3 = (32, 64, 128) if light_weight else (64, 128, 256)

        # Small CNN to process frequency magnitude map
        # Input is Magnitude map (1 channel)
        self.freq_cnn = nn.Sequential(
            ConvBnRelu(1, c1, kernel_size=3, stride=2, padding=1), # H/2, W/2
            ConvBnRelu(c1, c2, kernel_size=3, stride=2, padding=1), # H/4, W/4
            ConvBnRelu(c2, c3, kernel_size=3, stride=2, padding=1), # H/8, W/8
            ConvBnRelu(c3, embed_dim, kernel_size=3, stride=2, padding=1) # H/16, W/16 -> Should match target_grid_hw
            # Adaptive pooling to ensure exact target size, handles variations
            # nn.AdaptiveAvgPool2d((self.target_grid_hw, self.target_grid_hw)) # Optional: Use if strides don't perfectly match target
        )
        # Final 1x1 conv to potentially adjust embed_dim if needed (or could be part of freq_cnn)
        # self.final_conv = nn.Conv2d(c3, embed_dim, kernel_size=1) # Moved into Sequential above

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image tensor (B, C, H, W).
        Returns:
            torch.Tensor: Frequency feature map (B, embed_dim, target_grid_hw, target_grid_hw).
        """
        # Ensure input has the expected number of channels (handle grayscale/RGB)
        if x.size(1) == 1 and self.in_channels == 3:
            x = x.repeat(1, 3, 1, 1) # Repeat grayscale to 3 channels if model expects 3
        elif x.size(1) == 3 and self.in_channels == 1:
            # Convert RGB to grayscale (e.g., average or weighted sum)
            # Using simple average here
            x = torch.mean(x, dim=1, keepdim=True)

        # --- FFT Calculation ---
        # Calculate FFT (using rfft2 for real inputs, slightly more efficient)
        fft_raw = torch.fft.fft2(x, dim=(-2, -1)) # (B, C, H, W) complex tensor
        fft_shifted = torch.fft.fftshift(fft_raw, dim=(-2, -1)) # Shift zero freq to center

        # Calculate Magnitude Spectrum
        magnitude_spectrum = torch.abs(fft_shifted) # (B, C, H, W)

        # Average magnitude across input channels if C > 1
        if magnitude_spectrum.size(1) > 1:
             magnitude_spectrum = torch.mean(magnitude_spectrum, dim=1, keepdim=True) # (B, 1, H, W)

        # Log scale for stability (optional but common)
        magnitude_spectrum_log = torch.log1p(magnitude_spectrum)

        # --- CNN Processing ---
        freq_features = self.freq_cnn(magnitude_spectrum_log) # (B, embed_dim, H/16, W/16)

        # --- Ensure Target Grid Size ---
        # Use interpolation if the CNN output size doesn't match exactly
        current_h, current_w = freq_features.shape[-2:]
        if current_h != self.target_grid_hw or current_w != self.target_grid_hw:
            freq_features = F.interpolate(freq_features,
                                          size=(self.target_grid_hw, self.target_grid_hw),
                                          mode='bilinear', # or 'nearest'
                                          align_corners=False)
            # print(f"Warning: FSB output size {current_h}x{current_w} != target {self.target_grid_hw}x{self.target_grid_hw}. Resized.") # Debugging


        # freq_features = self.final_conv(freq_features) # Now part of Sequential

        return freq_features

# --- End of Frequency Supplement Block ---
# --- Add this block to a new file (e.g., attention_module.py) or within vit_seg_modeling.py ---

class SpatialAttentionFusion(nn.Module):
    """
    Fuses spatial and frequency features using a simple spatial attention mechanism.
    Assumes both features have the same spatial dimensions (H_patch, W_patch)
    and the same embedding dimension (embed_dim).
    """
    def __init__(self, embed_dim, reduction=16, kernel_size=7):
        """
        Args:
            embed_dim (int): Channel dimension of input features.
            reduction (int): Reduction ratio for the channel attention part (if added).
            kernel_size (int): Kernel size for the spatial attention convolution.
        """
        super().__init__()
        self.embed_dim = embed_dim

        # Simple Fusion: Concatenate -> Conv1x1 to fuse channels
        # This is often a good starting point before complex attention.
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        # Optional: Add Spatial Attention after fusion
        # assert kernel_size % 2 == 1, "Kernel size must be odd for spatial attention"
        # self.spatial_attn = nn.Sequential(
        #     nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False), # Use avg + max pool
        #     nn.Sigmoid()
        # )

    def forward(self, spatial_features, freq_features):
        """
        Args:
            spatial_features (torch.Tensor): High-level spatial features from ViT
                                             (B, embed_dim, H_patch, W_patch).
            freq_features (torch.Tensor): Frequency features from FSB
                                          (B, embed_dim, H_patch, W_patch).
        Returns:
            torch.Tensor: Fused feature map (B, embed_dim, H_patch, W_patch).
        """
        # Ensure features are on the same device
        if spatial_features.device != freq_features.device:
            freq_features = freq_features.to(spatial_features.device)

        # --- Method 1: Simple Concatenation + 1x1 Convolution Fusion ---
        combined_features = torch.cat([spatial_features, freq_features], dim=1) # (B, 2*embed_dim, H, W)
        fused_features = self.fusion_conv(combined_features) # (B, embed_dim, H, W)

        # --- Method 2: Addition (Simplest) ---
        # fused_features = spatial_features + freq_features # Element-wise addition

        # --- Method 3: Concatenation + Spatial Attention (More complex) ---
        # combined_features = torch.cat([spatial_features, freq_features], dim=1)
        # fused_simple = self.fusion_conv(combined_features) # Get initial fusion
        #
        # # Calculate Spatial Attention Map
        # avg_pool = torch.mean(fused_simple, dim=1, keepdim=True) # Avg along channel
        # max_pool, _ = torch.max(fused_simple, dim=1, keepdim=True) # Max along channel
        # pool_cat = torch.cat([avg_pool, max_pool], dim=1) # (B, 2, H, W)
        # attn_map = self.spatial_attn(pool_cat) # (B, 1, H, W)
        #
        # # Apply attention
        # fused_features = fused_simple * attn_map # Element-wise multiplication

        return fused_features

# --- End of Attention Aggregation Module ---