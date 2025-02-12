import torch
from torch import nn

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device} device")


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
        x += self.pos_embedding

        return x

class Encoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

encoder = Encoder().to(device)
patch_embedder = PatchEmbedder()

if __name__ == "__main__":
    print(encoder)

    batch_size = 4
    sample_images = torch.randn(batch_size, 1, 56, 56)
    output = patch_embedder(sample_images)
    print(f"Input shape: {sample_images.shape}")
    print(f"Output shape: {output.shape}") 
