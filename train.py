from torch import nn, optim
from transformer import VisionTransformer
from dataloader import four_digit_train

model = VisionTransformer()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.data)
    print(size)


print('training...')
train(four_digit_train, model, loss_fn, optimizer)
