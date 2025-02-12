from torch import nn, optim
from model import encoder

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-5)

print('training...')