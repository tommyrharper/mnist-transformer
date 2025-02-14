import torch
from torch import nn
from src2.models import PatchEmbedder, Encoder, Decoder

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

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

        self.encoders = nn.ModuleList([
            Encoder(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])

        # Separate classifier for each digit position
        self.digit_classifier = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, 10)  # outputs logits for digits 0-9
        )
        
        self.position_classifier = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, 4)  # outputs logits for positions 0-3
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        x = self.patch_embedder(x)
        
        # Prepend CLS token to sequence
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.norm(x)
        x = self.dropout(x)
        
        for encoder in self.encoders:
            x = encoder(x)
        
        # Use CLS token for classification
        x = x[:, 0]  # take first token [batch_size, embed_dim]
        
        # Return logits instead of argmax
        digit_logits = self.digit_classifier(x)      # [batch_size, 10]
        position_logits = self.position_classifier(x) # [batch_size, 4]
        
        return digit_logits, position_logits

if __name__ == "__main__":
    transformer = VisionTransformer()

    batch_size = 4
    sample_images = torch.randn(batch_size, 56, 56)

    digit_logits, position_logits = transformer(sample_images)
    print(f"Input shape: {sample_images.shape}")
    print(f"Output digit shape: {digit_logits.shape}") 
    print(f"Output digit: {digit_logits[0]}")
    print(f"Output digit: {digit_logits[1]}")
    print(f"Output digit: {digit_logits[2]}")
    print(f"Output digit: {digit_logits[3]}")
    print(f"Output position shape: {position_logits.shape}") 
    print(f"Output position: {position_logits[0]}")
    print(f"Output position: {position_logits[1]}")
    print(f"Output position: {position_logits[2]}")
    print(f"Output position: {position_logits[3]}")
