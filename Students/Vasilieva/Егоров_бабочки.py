import torch
from torch import nn
import torchvision as tv
import torchvision.models as models
import pandas as pd
import os
from PIL import Image

BATCH_SIZE = 64

# Предобработка изображений
transforms = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        for file_name in os.listdir(root_dir):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')):
                self.image_paths.append(os.path.join(root_dir, file_name))
        self.image_paths.sort()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        file_name = os.path.basename(img_path)  # "0.jpg" или "100.jpg"
        img_index = int(os.path.splitext(file_name)[0])  # "0" -> 0, "100" -> 100
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_index

# Загружаем данные
train_dataset = tv.datasets.ImageFolder(
    root="train_butterflies/train_split",
    transform=transforms
)

test_dataset = TestDataset(
    root_dir="test_butterflies/valid",
    transform=val_transforms
)

# Создаем загрузчики
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.Adam(model.classifier[1].parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss_val = loss_fn(y_pred, y)
        loss_val.backward()
        optimizer.step()
        running_loss += loss_val.item()
    
    print(f"Эпоха {epoch+1}/10, Loss: {running_loss/len(train_iter):.4f}")

# Размораживаем все слои для тонкой настройки
for param in model.parameters():
    param.requires_grad = True

# Используем меньшую скорость обучения для тонкой настройки
optimizer_finetune = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(5):
    model.train()
    running_loss = 0.0
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        optimizer_finetune.zero_grad()
        y_pred = model(X)
        loss_val = loss_fn(y_pred, y)
        loss_val.backward()
        optimizer_finetune.step()
        running_loss += loss_val.item()
    
    print(f"Эпоха {epoch+1}/5, Loss: {running_loss/len(train_iter):.4f}")

model.eval()
all_predictions_finetune = []
image_indices_finetune = []

with torch.no_grad():
    for X, indices in test_iter:
        X = X.to(device)
        logits = model(X)
        batch_predictions = torch.argmax(logits, dim=1)
        all_predictions_finetune.extend(batch_predictions.cpu().numpy())

        if torch.is_tensor(indices):
            indices_list = indices.cpu().numpy().tolist()
        else:
            indices_list = list(indices)

        indices_list = [int(idx) for idx in indices_list]
        image_indices_finetune.extend(indices_list)

# Конвертируем в номера классов
real_classes_finetune = []
for pred_idx in all_predictions_finetune:
    folder_name = train_dataset.classes[pred_idx]
    class_num = int(folder_name.replace('class_', ''))
    real_classes_finetune.append(class_num)

results_finetune = pd.DataFrame({
    'index': image_indices_finetune,
    'label': real_classes_finetune
})
results_finetune = results_finetune.sort_values(by='index')
results_finetune.to_csv('submission_efficientnet_finetune.csv', index=False)
print("Сохранён файл submission_efficientnet_finetune.csv")