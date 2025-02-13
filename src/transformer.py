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

        self.blocks = nn.ModuleList([
            *[Encoder(embed_dim, num_heads, ff_dim) for _ in range(num_layers)],
            *[Decoder(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        ])

        self.digit_positions = nn.Parameter(torch.randn(4, embed_dim) * 0.02)

        self.classifier = nn.Linear(embed_dim, 10)

    def forward(self, x):
        encoded = self.patch_embedder(x)

        for block in self.blocks[:len(self.blocks)//2]:
            encoded = block(encoded)

        batch_size = x.shape[0]
        decoded = self.digit_positions.unsqueeze(0).expand(batch_size, -1, -1)

        for block in self.blocks[len(self.blocks)//2:]:
            decoded = block(decoded, encoded)

        digits = []
        for i in range(4):
            digit = self.classifier(decoded[:, i])
            digits.append(digit)

        return digits

if __name__ == "__main__":
    print('transformers bro')
