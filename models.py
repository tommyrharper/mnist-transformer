import torch
from torch import nn

class PatchEmbedder(nn.Module):
    def __init__(self, image_size=56, patch_size=7, embed_dim=256):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size ** 2

        self.projection = nn.Linear(patch_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.rand(1, self.num_patches, embed_dim))

    def forward(self, x):
#         # Input: (batch_size, channels=1, height=56, width=56)
        batch_size = x.shape[0]

        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, self.num_patches, self.patch_size ** 2)

        x = self.projection(patches)
        x = x + self.pos_embedding

        return x

class Encoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, ff_dim=512):
        super().__init__()

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        attention_output, _ = self.attention(x, x, x)
        x = x + attention_output
        x = self.norm1(x)

        ff_output = self.ff(x)
        x = x +ff_output
        x = self.norm2(x)

        return x

class Decoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, ff_dim=512):
        super().__init__()

        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x, encoder_output):
        self_attention_output, _ = self.self_attention(x, x, x)
        x = x + self_attention_output
        x = self.norm1(x)

        cross_attention_output, _ = self.cross_attention(x, encoder_output, encoder_output)
        x = x + cross_attention_output
        x = self.norm2(x)

        ff_output = self.ff(x)
        x = x + ff_output
        x = self.norm3(x)

        return x

if __name__ == "__main__":
    encoder = Encoder()
    decoder = Decoder()
    patch_embedder = PatchEmbedder()
    print(encoder)
    print(decoder)
    print(patch_embedder)

    batch_size = 4
    sample_images = torch.randn(batch_size, 1, 56, 56)
    output = patch_embedder(sample_images)
    print(f"Input shape: {sample_images.shape}")
    print(f"Output shape: {output.shape}") 
