from torch import nn, optim
from model import encoder
from dataloader import four_digit_train, four_digit_test

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-5)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.data)
    print(size)


print('training...')
train(four_digit_train, encoder, loss_fn, optimizer)
