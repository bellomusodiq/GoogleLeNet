import math

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class GrayScaleTransform:
    def __call__(self, x):
        if x.size(0) == 1:
            x = torch.cat((x, x, x), dim=0)
        return x


dataset = datasets.Caltech256(
    root="./data",
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((224, 224)), GrayScaleTransform()]
    ),
    download=True,
)

train_data_len = math.floor(len(dataset) * 0.8)
test_data_len = len(dataset) - train_data_len
train_data, test_data = torch.utils.data.random_split(
    dataset, [train_data_len, test_data_len]
)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
