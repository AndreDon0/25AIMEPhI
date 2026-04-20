import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
import time
from torch.utils.data import DataLoader

BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transforms = tv.transforms.Compose([
    tv.transforms.Resize(32),
    tv.transforms.RandomRotation(10),   # Аугментация
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = tv.datasets.MNIST('.',
                                  train=True,
                                  transform=transforms,
                                  download=True)

test_dataset = tv.datasets.MNIST('.',
                                 train=False,
                                 download=True)

test_transforms = tv.transforms.Compose([
    tv.transforms.Resize(32),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset.transform = test_transforms

train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class ModelForMNIST(nn.Module):
    def __init__(self):
        super(ModelForMNIST, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=0)

        self.fc1 = nn.Linear(128, 84)
        self.fc2 = nn.Linear(84, 10)

        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

model = ModelForMNIST().to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

criterion = nn.CrossEntropyLoss()

def evaluate_accuracy(data_iter, net, device):
    net.eval()
    acc_sum, n = 0, 0

    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            acc_sum += (y_hat.argmax(axis=1) == y).sum().item()
            n += y.shape[0]

    net.train()
    return acc_sum / n

def train(net, train_iter, test_iter, optimizer, scheduler, num_epochs, device):
    best_acc = 0.0

    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()

        net.train()

        for X, y in train_iter:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            y_hat = net(X)
            l = criterion(y_hat, y)
            l.backward()
            optimizer.step()
            # w_i+1  = w_i - lr * grad(U_пот) 

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().item()
            n += y.shape[0]

        test_acc = evaluate_accuracy(test_iter, net, device)

        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc

        print(f'Epoch {epoch + 1:2d} | '
              f'Loss: {train_l_sum / n:.4f} | '
              f'Train Acc: {train_acc_sum / n:.3f} | '
              f'Test Acc: {test_acc:.3f} | '
              f'Time: {time.time() - start:.1f}s | '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')

    print(f'Лучшая точность на тесте: {best_acc:.4f}')
    return best_acc

best_accuracy = train(model, train_iter, test_iter, optimizer, scheduler, NUM_EPOCHS, DEVICE)

print(f"Результат: {best_accuracy:.4f}")
