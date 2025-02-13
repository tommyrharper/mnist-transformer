import torch
from torch import nn
from src.models import PatchEmbedder, Encoder, Decoder

class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size=56,
                 patch_size=7,
                 embed_dim=256,
                 num_heads=8,
                 num_layers=6,
                 ff_dim=512):
        super().__init__()
        self.num_layers = num_layers

        self.patch_embedder = PatchEmbedder(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim
        )

        self.encoders = nn.ModuleList([
            Encoder(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])

        # Separate classifier for each digit position
        self.digit_classifiers = nn.ModuleList([
            nn.Linear(embed_dim, 10) for _ in range(4)
        ])

    def forward(self, x):
        x = self.patch_embedder(x)
        
        # Encode patches
        for encoder in self.encoders:
            x = encoder(x)
        
        # Use different regions for different digits
        # Assuming 8x8 grid of patches
        tl = x[:, :16].mean(dim=1)  # top-left region
        tr = x[:, 16:32].mean(dim=1)  # top-right region
        bl = x[:, 32:48].mean(dim=1)  # bottom-left region
        br = x[:, 48:].mean(dim=1)  # bottom-right region
        
        # Predict each digit separately
        digits = [
            self.digit_classifiers[0](tl),
            self.digit_classifiers[1](tr),
            self.digit_classifiers[2](bl),
            self.digit_classifiers[3](br)
        ]
        
        return digits

if __name__ == "__main__":
    print('transformers bro')
