import torch

from .dataset import test_loader, train_loader
from .epoch_loops import epoch_test, epoch_train
from .loss import InceptionLoss
from .model import InceptionNet

train_losses = []
test_losses = []
test_accuracies = []

num_epochs = 50
learning_rage = 0.001

path = "./inception.pt"

CUDA = torch.cuda.is_available()

model = InceptionNet()
if CUDA:
    model = model.cuda()

loss_fn = InceptionLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == __main__:
    for epoch in range(num_epochs):
        epoch_train(model, num_epochs, epoch, train_loader, loss_fn, optimizer)
        epoch_test(model, num_epochs, epoch, test_loader, loss_fn)
        torch.save(model.state_dict(), path)
